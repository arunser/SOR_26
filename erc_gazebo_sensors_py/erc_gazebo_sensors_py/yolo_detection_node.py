import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time

from ultralytics import YOLO


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # Better model than yolov8n
        self.model = YOLO("yolov8s.pt")

        self.get_logger().info("YOLO model loaded")

        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1
        )

        self.bridge = CvBridge()

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.running = True

        self.spin_thread = threading.Thread(
            target=self.spin_thread_func,
            daemon=True
        )
        self.spin_thread.start()

        self.prev_time = time.time()

    def spin_thread_func(self):

        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        with self.frame_lock:
            self.latest_frame = frame

    def stop(self):

        self.running = False

        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)

    def display_image(self):

        cv2.namedWindow(
            "YOLO Detection",
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
        )

        cv2.resizeWindow("YOLO Detection", 1600, 900)

        while rclpy.ok() and self.running:

            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:

                result = self.run_yolo(frame)

                cv2.imshow("YOLO Detection", result)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()

    def run_yolo(self, frame):

        CONF_THRESHOLD = 0.35
        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=640,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                detections.append(
                    f"{class_name} ({confidence:.2f})"
                )
                color = self.class_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2
                )
                label = f"{class_name} {confidence:.2f}"

                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                text_y = max(y1 - 10, th + 10)

                cv2.rectangle(frame, (x1, text_y - th - baseline), (x1 + tw + 10, text_y + baseline), color, -1
                )
                cv2.putText(frame, label, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, color, -1
                )
                
        current_time = time.time()
        fps = 1.0 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time
        dashboard_width = 350
        dashboard = np.zeros(
            (frame.shape[0], dashboard_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            dashboard, "Detections", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )

        cv2.putText(dashboard,f"FPS : {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.putText(dashboard, f"Objects : {len(detections)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        y = 170

        for det in detections[:25]:

            cv2.putText(dashboard, det, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

            y += 30

        combined = np.hstack((frame, dashboard))

        return combined

    def class_color(self, class_id):

        np.random.seed(class_id)

        return tuple(
            int(c)
            for c in np.random.randint(100, 255, 3)
        )


def main(args=None):

    print("OpenCV Version:", cv2.__version__)

    rclpy.init(args=args)

    node = YoloDetectorNode()

    try:
        node.display_image()

    except KeyboardInterrupt:
        pass

    finally:

        node.stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()