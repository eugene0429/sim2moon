#!/usr/bin/env python3
"""Spawn or teleport a rover via ROS2 topics.

Usage:
    # Teleport existing rover to position
    python3 tools/spawn_rover.py --name pragyaan --pos 10 10 0.5

    # Teleport with orientation (quaternion x y z w)
    python3 tools/spawn_rover.py --name pragyaan --pos 10 10 0.5 --quat 0 0 0.707 0.707

    # Spawn new rover from USD
    python3 tools/spawn_rover.py --spawn --name husky \
        --usd assets/USD_Assets/robots/ros2_husky_PhysX_vlp16.usd \
        --pos 5 5 0.5

    # Reset rover to initial pose
    python3 tools/spawn_rover.py --reset --name pragyaan

    # Reset all rovers
    python3 tools/spawn_rover.py --reset-all
"""

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Empty


def main():
    parser = argparse.ArgumentParser(description="Spawn/teleport/reset rover")
    parser.add_argument("--name", default="pragyaan", help="Robot name")
    parser.add_argument("--pos", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=[10.0, 10.0, 0.5], help="Position x y z")
    parser.add_argument("--quat", type=float, nargs=4, metavar=("X", "Y", "Z", "W"),
                        default=[0.0, 0.0, 0.0, 1.0], help="Orientation quaternion x y z w")
    parser.add_argument("--spawn", action="store_true", help="Spawn new rover (requires --usd)")
    parser.add_argument("--usd", default="", help="USD path for spawn")
    parser.add_argument("--reset", action="store_true", help="Reset rover to initial pose")
    parser.add_argument("--reset-all", action="store_true", help="Reset all rovers")
    args = parser.parse_args()

    rclpy.init()
    node = Node("spawn_rover")

    qos = QoSProfile(depth=10,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)

    if args.reset_all:
        pub = node.create_publisher(Empty, "/OmniLRS/Robots/ResetAll", qos)
        rclpy.spin_once(node, timeout_sec=0.5)
        pub.publish(Empty())
        rclpy.spin_once(node, timeout_sec=0.5)
        print(f"Reset all rovers")

    elif args.reset:
        pub = node.create_publisher(String, "/OmniLRS/Robots/Reset", qos)
        rclpy.spin_once(node, timeout_sec=0.5)
        msg = String()
        msg.data = args.name
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.5)
        print(f"Reset '{args.name}'")

    elif args.spawn:
        if not args.usd:
            parser.error("--spawn requires --usd path")
        pub = node.create_publisher(PoseStamped, "/OmniLRS/Robots/Spawn", qos)
        rclpy.spin_once(node, timeout_sec=0.5)
        msg = PoseStamped()
        msg.header.frame_id = f"{args.name}:{args.usd}"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = args.pos
        msg.pose.orientation.x, msg.pose.orientation.y = args.quat[0], args.quat[1]
        msg.pose.orientation.z, msg.pose.orientation.w = args.quat[2], args.quat[3]
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.5)
        print(f"Spawn '{args.name}' at {args.pos}")

    else:
        pub = node.create_publisher(PoseStamped, "/OmniLRS/Robots/Teleport", qos)
        rclpy.spin_once(node, timeout_sec=0.5)
        msg = PoseStamped()
        msg.header.frame_id = args.name
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = args.pos
        msg.pose.orientation.x, msg.pose.orientation.y = args.quat[0], args.quat[1]
        msg.pose.orientation.z, msg.pose.orientation.w = args.quat[2], args.quat[3]
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.5)
        print(f"Teleport '{args.name}' to {args.pos}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
