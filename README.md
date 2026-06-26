An autonomous object-seeking behaviour for a differential-drive robot in Gazebo,
built on top of a YOLOv8 perception node. Given the name of an object, the robot
searches its surroundings, locks onto the target, drives toward it, and stops at a
safe distance.

The pipeline runs as a single state machine, updated on every camera frame:

```
SEARCH  ->  DETECT  ->  TRACK  ->  APPROACH  ->  COMPLETE
```

## Features

- **Target selection at runtime** — type any COCO class (e.g. `person`,
  `fire hydrant`, `refrigerator`) in the terminal; the robot focuses on that class
  and ignores all other detections.
- **Depth-based distance estimation** — reads the depth image at the target's
  bounding-box centre, taking the median over a small patch and rejecting invalid
  pixels for a stable reading.
- **Active search** — rotates in place to scan the room until the target appears.
- **Tracking and approach** — centres the target, then drives forward at a speed
  proportional to the remaining distance.
- **Safe stop** — halts and publishes zero velocity at a fixed safe distance, then
  waits for the next target without restarting.
- **Mission-control dashboard** — live overlay of target, status, distance, and mode.

## Requirements

- ROS 2 (tested on Jazzy, Ubuntu 24.04)
- Gazebo Sim with the `erc_gazebo_sensors` package
- Python packages: `ultralytics`, `opencv-python`, `numpy`, `cv_bridge`

