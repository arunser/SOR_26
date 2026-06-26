#!/usr/bin/env python3

import time
import signal
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from ultralytics import YOLO


CONF_THRESHOLD   = 0.25     # YOLO confidence threshold (low: we filter by class, so low-confidence non-target detections are ignored)


KP_ANGULAR       = 0.9      # turn gain, applied to normalised horizontal error
KP_LINEAR        = 0.4      # forward gain, applied to (distance - stop_distance)
MAX_ANGULAR      = 0.7      # rad/s  cap on turning
MAX_LINEAR       = 0.28     # m/s    cap on forward speed
SEARCH_ANGULAR   = 0.35     # rad/s  spin speed while searching (slow = more frames on target during a sweep -> easier to lock)

CENTER_TOL       = 0.12     # |error| (fraction of half-width) considered "centred"
STOP_DISTANCE    = 0.70     # m  -- safe stopping distance (see report)
HOLD_TIME        = 1.5      # s  -- coast on last detection this long before searching

# A target taller/wider than the robot OVERFILLS the camera as we close in, so
# YOLO can no longer recognise it. Once we are close/centred we LATCH a commit
# and finish on depth alone, instead of treating the dropout as "lost".
NEAR_COMMIT_DIST = 1.30     # m  commit to final approach once this close
NEAR_COMMIT_FRAC = 0.60     # or once the target box fills this fraction of height
FINAL_FWD        = 0.10     # m/s slow creep during the blind final approach
FINAL_LOST_DIST  = 2.50     # m  if centre depth exceeds this, the commit was wrong
FINAL_TIMEOUT    = 4.0      # s  give up a blind approach after this long

# depth sampling
DEPTH_PATCH      = 5        # half-size: a (2*5+1)=11 x 11 window around the centre
DEPTH_MIN        = 0.30     # m  camera near clip (from .gazebo)
DEPTH_MAX        = 15.0     # m  camera far clip
MIN_VALID_PIXELS = 10       # need at least this many good pixels for a reading


#  Mission states                                                               
class State:
    WAIT     = "WAITING"     # no target chosen yet
    SEARCH   = "SEARCHING"   # spinning, looking for the target
    TRACK    = "TRACKING"    # target seen, centring it
    APPROACH = "APPROACH"    # centred, driving forward
    DONE     = "REACHED"     # mission complete


class ObjectHuntNode(Node):

    def __init__(self):
        super().__init__('object_hunt')

        self.model = YOLO("yolov8s.pt")
        self.class_names = self.model.names                 
        self.valid_classes = {n.lower() for n in self.class_names.values()}
        self.get_logger().info("YOLO model loaded")

        self.bridge = CvBridge()
        self.create_subscription(Image, 'camera/image',       self.image_callback, 1)
        self.create_subscription(Image, 'camera/depth_image', self.depth_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.lock         = threading.Lock()
        self.latest_frame = None
        self.depth_image  = None
        self._depth_logged = False

        self.target_class = None
        self.state        = State.WAIT
        self.distance     = None
        self.lost_count   = 0
        self.last_target      = None     # last fresh detection (cx,cy,x1,y1,x2,y2)
        self.last_target_time = 0.0      # when we last saw it (for coasting)
        self.last_error       = 0.0      # last horizontal error (for blind heading)
        self.committed        = False    # latched: finish on depth even if YOLO drops
        self.commit_time      = 0.0      # when the current blind approach began
        self._blind           = False    # currently in blind final approach (display)
        self.running      = True

        self._last_search_log = 0.0
        self.prev_time = time.time()

        self.spin_thread = threading.Thread(target=self.spin_thread_func, daemon=True)
        self.spin_thread.start()

        self.input_thread = threading.Thread(target=self.input_thread_func, daemon=True)
        self.input_thread.start()


    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.lock:
            self.latest_frame = frame

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        arr = np.asarray(depth)

        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 1000.0
        else:
            arr = arr.astype(np.float32)
        with self.lock:
            self.depth_image = arr
        if not self._depth_logged:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                self.get_logger().info(
                    f"Depth OK: shape={arr.shape} encoding={msg.encoding} "
                    f"range=[{finite.min():.2f}, {finite.max():.2f}] m")
            else:
                self.get_logger().warn("Depth frame received but all pixels invalid")
            self._depth_logged = True


    def input_thread_func(self):
        time.sleep(1.0)
        while self.running:
            try:
                raw = input("\nEnter target object: ").strip().lower()
            except (EOFError, OSError):
                break
            if not self.running:
                break
            if raw in self.valid_classes:
                with self.lock:
                    self.target_class = raw
                    self.state        = State.SEARCH
                    self.distance     = None
                    self.lost_count   = 0
                    self.last_target  = None
                    self.committed    = False
                    self.commit_time  = 0.0
                print(f"Searching for: {raw}")
            elif raw:
                print(f"'{raw}' is not a known class. "
                      f"Try e.g.: person, bottle, chair, fire hydrant.")


    def get_distance(self, cx, cy):
        """Median depth (metres) over a small patch around (cx, cy)."""
        with self.lock:
            depth = None if self.depth_image is None else self.depth_image
            if depth is not None:
                depth = depth.copy()
        if depth is None:
            return None

        h, w = depth.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None

        k = DEPTH_PATCH
        x1, x2 = max(0, cx - k), min(w, cx + k + 1)
        y1, y2 = max(0, cy - k), min(h, cy + k + 1)
        patch = depth[y1:y2, x1:x2].reshape(-1)

        # reject NaN / inf and out-of-range pixels
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > DEPTH_MIN) & (valid < DEPTH_MAX)]
        if valid.size < MIN_VALID_PIXELS:
            return None
        return float(np.median(valid))


    def detect_target(self, frame):
        """
        Run YOLO, draw every detection, and return the best matching instance
        of the target class as (cx, cy, x1, y1, x2, y2) or None.
        'Best' = largest bounding box (nearest / most prominent instance).
        """
        results = self.model(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

        best = None
        best_area = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                name       = self.class_names[class_id]
                is_target  = (self.target_class is not None
                              and name.lower() == self.target_class)

                # focus only on the target: highlight it, dim everything else
                color = (0, 255, 0) if is_target else (140, 140, 140)
                thick = 3 if is_target else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                label = f"{name} {confidence:.2f}"
                cv2.putText(frame, label, (x1 + 3, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if is_target:
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        best = (cx, cy, x1, y1, x2, y2)

        # mark the locked target
        if best is not None:
            cx, cy, x1, y1, x2, y2 = best
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(frame, (cx, 0), (cx, frame.shape[0]), (0, 0, 255), 1)
        # vertical centre reference line
        midx = frame.shape[1] // 2
        cv2.line(frame, (midx, 0), (midx, frame.shape[0]), (255, 0, 0), 1)
        return best

    def update_control(self, frame, target):
        msg = Twist()
        h, w = frame.shape[:2]
        half = w / 2.0
        now = time.time()

        if self.state in (State.WAIT, State.DONE) or self.target_class is None:
            self.cmd_pub.publish(msg)
            return

        fresh = target is not None

        if fresh:
            if self.state == State.SEARCH:
                print("Target Found!")
            self.last_target = target
            self.last_target_time = now
            self.last_error = (target[0] - half) / half
            self.commit_time = 0.0               

        # short flicker dropout: coast on the last position (depth still live)
        coasting = (not fresh and self.last_target is not None
                    and (now - self.last_target_time) < HOLD_TIME
                    and self.state in (State.TRACK, State.APPROACH))

        # long dropout but we had committed (target now overfills the frame and
        # YOLO can't see it): keep finishing on depth -> "blind" final approach
        blind = (not fresh and not coasting and self.committed
                 and self.state in (State.TRACK, State.APPROACH))
        self._blind = blind

        if not fresh and not coasting and not blind:
            self.state = State.SEARCH
            self.distance = None
            self.committed = False
            self.commit_time = 0.0
            self.last_target = None
            msg.angular.z = SEARCH_ANGULAR        # spin in place to scan the room
            self.cmd_pub.publish(msg)
            if now - self._last_search_log > 1.0:
                print("Searching...")
                self._last_search_log = now
            return

        # ---- BLIND FINAL APPROACH : finish on depth, ignore the lost detection ----
        if blind:
            cx = int(np.clip(self.last_target[0], 0, w - 1))
            cy = int(np.clip(self.last_target[1], 0, h - 1))
            self.distance = self.get_distance(cx, cy)
            if self.distance is None:             # centre of frame is the object now
                self.distance = self.get_distance(w // 2, h // 2)
            if self.commit_time == 0.0:
                self.commit_time = now

            if self.distance is not None and self.distance <= STOP_DISTANCE:
                self._finish()
                return

            if (self.distance is not None and self.distance > FINAL_LOST_DIST) \
               or (now - self.commit_time > FINAL_TIMEOUT):
                self.state = State.SEARCH
                self.committed = False
                self.commit_time = 0.0
                self.last_target = None
                self.distance = None
                msg.angular.z = SEARCH_ANGULAR
                self.cmd_pub.publish(msg)
                return

            msg.linear.x = FINAL_FWD
            msg.angular.z = float(np.clip(-0.4 * self.last_error,
                                          -MAX_ANGULAR, MAX_ANGULAR))
            self.state = State.APPROACH
            self.cmd_pub.publish(msg)
            if self.distance is not None:
                print(f"Distance: {self.distance:.2f} m")
            return


        eff = target if fresh else self.last_target
        cx, cy = eff[0], eff[1]
        error = (cx - half) / half              
        self.distance = self.get_distance(cx, cy) 
        box_h_frac = (eff[5] - eff[3]) / float(h)

        if (self.distance is not None and self.distance <= NEAR_COMMIT_DIST) \
           or box_h_frac > NEAR_COMMIT_FRAC:
            self.committed = True

        reached = (self.distance <= STOP_DISTANCE) if self.distance is not None \
                  else (box_h_frac > 0.92)
        if reached:
            self._finish()
            return

        # centre first (TRACK), then drive forward (APPROACH)
        msg.angular.z = float(np.clip(-KP_ANGULAR * error, -MAX_ANGULAR, MAX_ANGULAR))
        if abs(error) < CENTER_TOL:
            self.state = State.APPROACH
            if self.distance is not None:
                fwd = KP_LINEAR * (self.distance - STOP_DISTANCE)
            else:
                fwd = 0.10                         # creep if depth momentarily missing
            if not fresh:
                fwd *= 0.5                         # be cautious while coasting
            msg.linear.x = float(np.clip(fwd, 0.0, MAX_LINEAR))
        else:
            self.state = State.TRACK              # turn in place to centre target

        self.cmd_pub.publish(msg)
        if self.distance is not None:
            print(f"Distance: {self.distance:.2f} m")

    def _finish(self):
        """Stop the robot and end the mission (Stage 6)."""
        self.state = State.DONE
        self.committed = False
        self.commit_time = 0.0
        self._blind = False
        self.cmd_pub.publish(Twist())             # publish zero velocity
        print("\nMission Completed")
        print("Target Reached Successfully")

    # mission-control dashboard panel                         
  
    def draw_dashboard(self, frame):
        panel_w = 360
        panel = np.full((frame.shape[0], panel_w, 3), 25, dtype=np.uint8)

        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        target = self.target_class if self.target_class else "-"
        dist   = f"{self.distance:.2f} m" if self.distance is not None else "-- m"

        status_color = {
            State.WAIT:     (180, 180, 180),
            State.SEARCH:   (0, 200, 255),
            State.TRACK:    (0, 255, 255),
            State.APPROACH: (0, 165, 255),
            State.DONE:     (0, 255, 0),
        }.get(self.state, (255, 255, 255))

        def put(text, y, color=(255, 255, 255), scale=0.7, thick=2):
            cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick)

        put("MISSION CONTROL", 40, (0, 255, 255), 0.8, 2)
        cv2.line(panel, (20, 55), (panel_w - 20, 55), (60, 60, 60), 1)
        put(f"TARGET   : {target}",        100)
        put(f"STATUS   : {self.state}",    140, status_color)
        put(f"DISTANCE : {dist}",          180)
        put(f"MODE     : {self._mode()}",  220)
        put(f"FPS      : {fps:.1f}",       270, (0, 255, 0), 0.6, 1)
        put(f"STOP @   : {STOP_DISTANCE:.2f} m", 300, (160, 160, 160), 0.6, 1)

        if self.state == State.DONE:
            put("TARGET REACHED", 360, (0, 255, 0), 0.7, 2)
            put("Enter a new target", 395, (160, 160, 160), 0.5, 1)
            put("in the terminal.",   415, (160, 160, 160), 0.5, 1)

        return np.hstack((frame, panel))

    def _mode(self):
        if self.state == State.APPROACH and self._blind:
            return "FINAL"
        return {
            State.WAIT:     "IDLE",
            State.SEARCH:   "SCAN",
            State.TRACK:    "CENTERING",
            State.APPROACH: "APPROACH",
            State.DONE:     "COMPLETE",
        }.get(self.state, "-")

  
    def run(self):
        cv2.namedWindow("Object Hunt", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Object Hunt", 1500, 850)

        while rclpy.ok() and self.running:
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:
                target = self.detect_target(frame)   
                self.update_control(frame, target)   
                view = self.draw_dashboard(frame)    
                cv2.imshow("Object Hunt", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
                break

        try:
            self.cmd_pub.publish(Twist())            # safety: zero velocity
        except Exception:
            pass
        cv2.destroyAllWindows()

    def stop(self):
        """Idempotent, safe shutdown: stop the robot, end threads, close GUI."""
        self.running = False
        # publish zero velocity a few times so the command actually reaches
        try:
            for _ in range(3):
                self.cmd_pub.publish(Twist())
                time.sleep(0.02)
        except Exception:
            pass
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main(args=None):
    print("OpenCV Version:", cv2.__version__)
    rclpy.init(args=args)
    node = ObjectHuntNode()

    def handle_sigint(signum, frame):
        if node.running:
            node.get_logger().info("Ctrl+C received -- shutting down...")
        node.running = False
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        node.run()
    except KeyboardInterrupt:
        node.running = False
    finally:
        node.stop()                
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()