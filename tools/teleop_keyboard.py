#!/usr/bin/env python3
"""Keyboard teleoperation for rover via ROS2 cmd_vel.

Usage:
    # Single rover
    python3 tools/teleop_keyboard.py --topic /husky/cmd_vel

    # Dual rover (mirror mode) — same command to both rovers simultaneously
    python3 tools/teleop_keyboard.py --topic /husky/cmd_vel --mirror /husky_physx/cmd_vel

Controls:
    W / Up      : forward
    S / Down    : backward
    A / Left    : turn left
    D / Right   : turn right
    Q           : forward + turn left
    E           : forward + turn right
    Z           : backward + turn left
    C           : backward + turn right
    Space       : emergency stop
    ESC         : quit
"""

import argparse
import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Key code mappings
KEY_BINDINGS = {
    # (linear_x, angular_z)
    "w": (1.0, 0.0),
    "s": (-1.0, 0.0),
    "a": (0.0, 1.0),
    "d": (0.0, -1.0),
    "q": (1.0, 1.0),
    "e": (1.0, -1.0),
    "z": (-1.0, 1.0),
    "c": (-1.0, -1.0),
    " ": (0.0, 0.0),
}

# Arrow key escape sequences (after \x1b[)
ARROW_KEYS = {
    "A": (1.0, 0.0),   # Up
    "B": (-1.0, 0.0),  # Down
    "D": (0.0, 1.0),   # Left
    "C": (0.0, -1.0),  # Right
}

HELP_TEXT = """
------------------------------------------
  Keyboard Teleop for Lunar Rover
------------------------------------------
  W / Up     : forward
  S / Down   : backward
  A / Left   : turn left
  D / Right  : turn right
  Q          : forward-left
  E          : forward-right
  Z          : backward-left
  C          : backward-right
  Space      : stop
  ESC        : quit
------------------------------------------
"""


class TeleopKeyboard(Node):
    def __init__(
        self,
        topics: list[str],
        linear_speed: float,
        angular_speed: float,
    ):
        super().__init__("teleop_keyboard")
        self.pubs = []
        for topic in topics:
            pub = self.create_publisher(Twist, topic, 10)
            self.pubs.append(pub)
            self.get_logger().info(f"Publishing Twist on '{topic}'")
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.get_logger().info(
            f"Linear speed: {linear_speed:.2f} m/s, Angular speed: {angular_speed:.2f} rad/s"
        )

    def publish(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = linear_x * self.linear_speed
        msg.angular.z = angular_z * self.angular_speed
        for pub in self.pubs:
            pub.publish(msg)


def get_key(timeout: float = 0.1) -> str | None:
    """Read a single key press without blocking (non-canonical terminal mode)."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Could be an arrow key escape sequence
            if select.select([sys.stdin], [], [], 0.02)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[" and select.select([sys.stdin], [], [], 0.02)[0]:
                    ch3 = sys.stdin.read(1)
                    return f"\x1b[{ch3}"
            return "\x1b"
        return ch
    return None


def main():
    parser = argparse.ArgumentParser(description="Keyboard teleop for rover")
    parser.add_argument("--topic", default="/cmd_vel", help="Primary cmd_vel topic")
    parser.add_argument(
        "--mirror", nargs="+", default=[], metavar="TOPIC",
        help="Additional cmd_vel topics to mirror the same commands to "
             "(e.g., --mirror /husky_physx/cmd_vel)",
    )
    parser.add_argument("--linear", type=float, default=0.3, help="Max linear speed (m/s)")
    parser.add_argument("--angular", type=float, default=0.3, help="Max angular speed (rad/s)")
    args = parser.parse_args()

    topics = [args.topic] + args.mirror

    rclpy.init()
    node = TeleopKeyboard(topics, args.linear, args.angular)

    # Save and set terminal to raw mode
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    print(HELP_TEXT)
    if len(topics) == 1:
        print(f"  Topic: {topics[0]}")
    else:
        print(f"  Topics (mirror mode):")
        for t in topics:
            print(f"    - {t}")
    print(f"  Linear: {args.linear:.1f} m/s | Angular: {args.angular:.1f} rad/s\n")

    try:
        while rclpy.ok():
            key = get_key(0.1)
            if key is None:
                continue

            # ESC (standalone, not arrow sequence)
            if key == "\x1b":
                node.publish(0.0, 0.0)
                print("\nStopping rover. Bye!")
                break

            # Arrow keys
            if key.startswith("\x1b[") and len(key) == 3:
                arrow = key[2]
                if arrow in ARROW_KEYS:
                    lx, az = ARROW_KEYS[arrow]
                    node.publish(lx, az)
                continue

            # Regular keys
            lower = key.lower()
            if lower in KEY_BINDINGS:
                lx, az = KEY_BINDINGS[lower]
                node.publish(lx, az)

    except KeyboardInterrupt:
        node.publish(0.0, 0.0)
        print("\nInterrupted. Stopping rover.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
