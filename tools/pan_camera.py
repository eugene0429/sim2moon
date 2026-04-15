#!/usr/bin/env python3
"""Pan (rotate in place) the Isaac Sim perspective camera.

The camera stays at its current position and rotates its viewing direction.

Must be run inside Isaac Sim (Console or Script Editor).

Usage (Console):
    exec(open("/home/sim2real1/sim2moon/tools/pan_camera.py").read())

    # Pan right 90 degrees over 5 seconds
    pan_right(90, duration=5.0)

    # Pan left 45 degrees over 3 seconds
    pan_left(45, duration=3.0)

    # Full 360 rotation in 20 seconds
    pan_full(duration=20.0)

    # Instant jump 30 degrees
    pan_camera(total_angle=30.0, duration=0.0)
"""

import math

import omni.kit.app
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf


def _rotate_point_around_axis(point, center, angle_rad, axis=Gf.Vec3d(0, 0, 1)):
    """Rotate a point around a vertical axis (Z-up) passing through center."""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_x = center[0] + dx * cos_a - dy * sin_a
    new_y = center[1] + dx * sin_a + dy * cos_a
    return Gf.Vec3d(new_x, new_y, point[2])


def pan_camera(total_angle=90.0, duration=5.0):
    """Pan the perspective camera in place (camera position stays fixed).

    Args:
        total_angle: Total pan angle in degrees. Positive = pan left
                     (counter-clockwise from above). Negative = pan right.
        duration:    Time in seconds for the sweep. 0 = instant jump.
    """
    cam_state = ViewportCameraState()
    cam_pos = cam_state.position_world
    cam_tgt = cam_state.target_world

    total_rad = math.radians(total_angle)

    print(f"[pan_camera] pos: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")
    print(f"[pan_camera] current target: ({cam_tgt[0]:.2f}, {cam_tgt[1]:.2f}, {cam_tgt[2]:.2f})")
    print(f"[pan_camera] pan {total_angle}deg over {duration}s")

    if duration <= 0:
        new_tgt = _rotate_point_around_axis(cam_tgt, cam_pos, total_rad)
        cam_state.set_target_world(new_tgt, True)
        print("[pan_camera] done (instant)")
        return

    # Animated sweep using Kit update events (wall-clock based)
    import time as _time
    state = {"sub": None, "prev_angle": 0.0, "t0": _time.time()}

    def _on_update(e):
        elapsed = _time.time() - state["t0"]
        frac = min(elapsed / duration, 1.0)

        current_angle = total_rad * frac
        delta = current_angle - state["prev_angle"]
        state["prev_angle"] = current_angle

        cs = ViewportCameraState()
        tgt = cs.target_world
        cam = cs.position_world
        new_tgt = _rotate_point_around_axis(tgt, cam, delta)
        cs.set_target_world(new_tgt, True)

        if frac >= 1.0:
            state["sub"] = None
            print("[pan_camera] done")

    update_stream = omni.kit.app.get_app().get_update_event_stream()
    state["sub"] = update_stream.create_subscription_to_pop(
        _on_update, name="pan_camera_update"
    )


# ── Convenience shortcuts ───────────────────────────────────────────
def pan_left(angle=90.0, duration=5.0):
    """Pan left (counter-clockwise from above)."""
    pan_camera(total_angle=abs(angle), duration=duration)


def pan_right(angle=90.0, duration=5.0):
    """Pan right (clockwise from above)."""
    pan_camera(total_angle=-abs(angle), duration=duration)


def pan_full(duration=20.0):
    """Full 360 degree pan."""
    pan_camera(total_angle=360.0, duration=duration)
