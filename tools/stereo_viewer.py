#!/usr/bin/env python3
"""Stereo image viewer — subscribe to left/right camera topics and display side by side.

Usage:
    # Default topics
    python3 tools/stereo_viewer.py

    # Custom topics
    python3 tools/stereo_viewer.py --left /husky/left/image_raw --right /husky/right/image_raw

    # With depth overlay
    python3 tools/stereo_viewer.py --left /husky/left/image_raw --right /husky/right/depth

    # Resize output
    python3 tools/stereo_viewer.py --width 1280

Controls:
    S         : save current frame pair as PNG
    Space     : pause / resume
    Q / ESC   : quit
"""

import argparse
import sys
import threading
from datetime import datetime

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class StereoViewer(Node):
    def __init__(
        self,
        left_topic: str,
        right_topic: str,
        display_width: int,
        slop: float = 0.05,
    ):
        super().__init__("stereo_viewer")
        self.bridge = CvBridge()
        self.left_image: np.ndarray | None = None
        self.right_image: np.ndarray | None = None
        self.lock = threading.Lock()
        self.display_width = display_width
        self.paused = False
        self.sync_count = 0
        self.drop_count = 0
        self.left_topic = left_topic
        self.right_topic = right_topic

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Time-synchronized subscribers
        left_sub = message_filters.Subscriber(self, Image, left_topic, qos_profile=qos)
        right_sub = message_filters.Subscriber(self, Image, right_topic, qos_profile=qos)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub],
            queue_size=10,
            slop=slop,
        )
        self._sync.registerCallback(self._cb_synced)

        self.get_logger().info(f"Left  topic: {left_topic}")
        self.get_logger().info(f"Right topic: {right_topic}")
        self.get_logger().info(f"Time sync slop: {slop:.3f}s")

    def _to_bgr(self, msg: Image) -> np.ndarray:
        """Convert ROS Image to BGR numpy array, handling various encodings."""
        encoding = msg.encoding.lower()

        if encoding in ("32fc1", "32fc"):
            # Depth image — normalize to 8-bit grayscale then convert to BGR
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            valid = np.isfinite(depth)
            if valid.any():
                d_min = float(np.min(depth[valid]))
                d_max = float(np.max(depth[valid]))
                if d_max - d_min > 1e-6:
                    normalized = np.zeros_like(depth, dtype=np.float32)
                    normalized[valid] = (depth[valid] - d_min) / (d_max - d_min)
                else:
                    normalized = np.zeros_like(depth, dtype=np.float32)
                gray = (normalized * 255).astype(np.uint8)
            else:
                gray = np.zeros((msg.height, msg.width), dtype=np.uint8)
            return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        if "mono" in encoding or encoding in ("8uc1",):
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _cb_synced(self, left_msg: Image, right_msg: Image):
        """Called only when left and right messages are time-synchronized."""
        with self.lock:
            self.left_image = self._to_bgr(left_msg)
            self.right_image = self._to_bgr(right_msg)
            self.sync_count += 1

    def get_stereo_frame(self) -> np.ndarray | None:
        """Compose side-by-side stereo frame. Returns None if no synced pair received."""
        with self.lock:
            left = self.left_image
            right = self.right_image

        if left is None or right is None:
            return None

        # Resize to same height
        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        target_h = max(lh, rh)
        if lh != target_h:
            scale = target_h / lh
            left = cv2.resize(left, (int(lw * scale), target_h))
        if rh != target_h:
            scale = target_h / rh
            right = cv2.resize(right, (int(rw * scale), target_h))

        # Separator line
        sep = np.full((target_h, 2, 3), 128, dtype=np.uint8)
        combined = np.hstack([left, sep, right])

        # Add labels
        label_bg = (0, 0, 0)
        cv2.rectangle(combined, (0, 0), (120, 28), label_bg, -1)
        cv2.putText(combined, "LEFT", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        rx_offset = left.shape[1] + 2
        cv2.rectangle(combined, (rx_offset, 0), (rx_offset + 130, 28), label_bg, -1)
        cv2.putText(combined, "RIGHT", (rx_offset + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Resize to display width
        ch, cw = combined.shape[:2]
        if self.display_width > 0 and cw != self.display_width:
            scale = self.display_width / cw
            combined = cv2.resize(combined, (self.display_width, int(ch * scale)))

        # Status bar
        status = f"Synced:{self.sync_count}"
        if self.paused:
            status += "  [PAUSED]"
        ch2, cw2 = combined.shape[:2]
        cv2.rectangle(combined, (0, ch2 - 24), (cw2, ch2), (0, 0, 0), -1)
        cv2.putText(combined, status, (8, ch2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return combined

    def save_frame(self, frame: np.ndarray):
        """Save current stereo frame to file."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"stereo_{ts}.png"
        cv2.imwrite(filename, frame)
        self.get_logger().info(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Stereo image viewer for ROS2 camera topics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--left", default="/stereo/left/rgb",
        help="Left camera topic (default: /left/image_raw)",
    )
    parser.add_argument(
        "--right", default="/stereo/right/rgb",
        help="Right camera topic (default: /right/image_raw)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Display width in pixels (default: 1280, 0=original)",
    )
    parser.add_argument(
        "--slop", type=float, default=0.05,
        help="Max time difference (sec) between left/right for sync (default: 0.05)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = StereoViewer(args.left, args.right, args.width, slop=args.slop)

    # Spin ROS2 in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    window_name = f"Stereo: {args.left} | {args.right}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n{'=' * 50}")
    print(f"  Stereo Viewer")
    print(f"  Left:  {args.left}")
    print(f"  Right: {args.right}")
    print(f"{'=' * 50}")
    print(f"  S     : save frame")
    print(f"  Space : pause / resume")
    print(f"  Q/ESC : quit")
    print(f"{'=' * 50}\n")

    last_frame = None

    try:
        while rclpy.ok():
            if not node.paused:
                frame = node.get_stereo_frame()
                if frame is not None:
                    last_frame = frame

            if last_frame is not None:
                cv2.imshow(window_name, last_frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("s") and last_frame is not None:
                node.save_frame(last_frame)
            elif key == ord(" "):
                node.paused = not node.paused
                state = "PAUSED" if node.paused else "RESUMED"
                node.get_logger().info(state)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
