"""Battery and solar power simulation model.

Tracks battery state-of-charge based on solar panel input and device loads.
Supports configurable device power draws, solar panel orientation,
battery voltage curve interpolation, and measurement noise.

Ported from OmniLRS src/robots/PowerModel.py with improved structure.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

# Device settings: name -> (off_power_w, on_power_w)
DEVICE_SETTINGS: Dict[str, Tuple[float, float]] = {
    "current_draw_obc": (0.0, 7.5),
    "current_draw_motor_controller": (0.0, 2.0),
    "current_draw_neutron_spectrometer": (0.0, 9.0),
    "current_draw_apxs": (0.0, 9.0),
    "current_draw_camera": (0.0, 5.0),
    "current_draw_radio": (0.0, 5.0),
    "current_draw_eps": (0.0, 1.0),
}

# Battery voltage curve: (state_of_charge_fraction, voltage)
BATTERY_VOLTAGE_CURVE: Tuple[Tuple[float, float], ...] = (
    (0.0, 11.0),
    (0.1, 12.0),
    (0.2, 13.5),
    (0.4, 15.0),
    (0.6, 15.8),
    (0.8, 16.3),
    (0.9, 16.6),
    (1.0, 16.8),
)

SOLAR_PANEL_NORMALS: Dict[str, np.ndarray] = {
    "deployed": np.array((0.0, 1.0, 0.0)),
    "stowed": np.array((0.0, 0.0, 1.0)),
}

BATTERY_CAPACITY_WH = 60.0
SOLAR_PANEL_MAX_POWER = 30.0
MOTOR_COUNT = 6
MOTOR_POWER_W = 10.0
REGULATED_BUS_VOLTAGE = 5.0
DC_DC_EFFICIENCY = 0.95
DEVICE_HEALTH_NOMINAL = "NOMINAL"
DEVICE_HEALTH_FAULT = "FAULT"
DEVICE_FAULT_EXTRA_POWER = 2.5


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class PowerModel:
    """Battery simulation with solar charging and device loads.

    Usage:
        1. Set inputs: rover position, yaw, sun position, device states
        2. Call step(dt) to advance simulation by dt seconds
        3. Read outputs: status(), battery_percentage(), battery_voltage()
    """

    battery_capacity_wh: float = BATTERY_CAPACITY_WH
    battery_charge_wh: float = BATTERY_CAPACITY_WH
    solar_panel_max_power: float = SOLAR_PANEL_MAX_POWER
    solar_panel_state: str = "stowed"
    motor_state: bool = False
    motor_count: int = MOTOR_COUNT
    motor_power_w: float = MOTOR_POWER_W

    device_power_settings: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEVICE_SETTINGS)
    )
    device_states: Dict[str, bool] = field(default_factory=dict)
    device_health: Dict[str, str] = field(default_factory=dict)

    device_current_noise_std: float = 0.02
    battery_percentage_noise_std: float = 0.5
    battery_voltage_noise_std: float = 0.05
    solar_input_noise_std: float = 0.1

    rover_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rover_yaw_deg: float = 0.0
    sun_position: Tuple[float, float, float] = (0.0, -10.0, 0.0)
    solar_input_power: float = 0.0
    battery_voltage_v: float = BATTERY_VOLTAGE_CURVE[-1][1]

    def __post_init__(self) -> None:
        for name in self.device_power_settings:
            self.device_states.setdefault(name, False)
            self.device_health.setdefault(name, DEVICE_HEALTH_NOMINAL)
        self.battery_charge_wh = _clamp(self.battery_charge_wh, 0.0, self.battery_capacity_wh)
        self._update_battery_voltage()

    # -- Input setters --

    def set_device_state(self, name: str, state: bool) -> None:
        if name not in self.device_power_settings:
            raise KeyError(f"Unknown device '{name}'")
        self.device_states[name] = state

    def set_device_states(self, states: Mapping[str, bool]) -> None:
        for name, state in states.items():
            self.set_device_state(name, state)

    def set_device_health(self, health_status: Mapping[str, str]) -> None:
        for name, status in health_status.items():
            if name not in self.device_power_settings:
                raise KeyError(f"Unknown device '{name}'")
            if status not in (DEVICE_HEALTH_NOMINAL, DEVICE_HEALTH_FAULT):
                raise ValueError(f"Invalid health '{status}' for device '{name}'")
            self.device_health[name] = status

    def set_rover_position(self, position: Tuple[float, float, float]) -> None:
        self.rover_position = position

    def set_rover_yaw(self, yaw_deg: float) -> None:
        self.rover_yaw_deg = yaw_deg

    def set_sun_position(self, position: Tuple[float, float, float]) -> None:
        self.sun_position = position

    def set_all_devices(self, state: bool) -> None:
        for name in self.device_states:
            self.device_states[name] = state

    def set_solar_panel_state(self, state: str) -> None:
        if state not in SOLAR_PANEL_NORMALS:
            raise ValueError(f"Unknown solar panel state '{state}', expected one of {list(SOLAR_PANEL_NORMALS)}")
        self.solar_panel_state = state

    def set_motor_state(self, state: bool) -> None:
        self.motor_state = state

    # -- Computation --

    def step(self, dt: float) -> None:
        """Advance the battery state by dt seconds."""
        view_factor = self.compute_view_factor(self.rover_position, self.sun_position)
        self.solar_input_power = self.solar_panel_max_power * view_factor
        net_power = self.solar_input_power - self.total_load_power()
        self.battery_charge_wh += net_power * (dt / 3600.0)
        self.battery_charge_wh = _clamp(self.battery_charge_wh, 0.0, self.battery_capacity_wh)
        self._update_battery_voltage()

    def compute_view_factor(
        self,
        rover_position: Tuple[float, float, float],
        sun_position: Tuple[float, float, float],
    ) -> float:
        """Compute solar panel view factor (0..1) based on sun direction."""
        rover = np.asarray(rover_position, dtype=float)
        sun = np.asarray(sun_position, dtype=float)
        vector = sun - rover
        magnitude = np.linalg.norm(vector)
        if magnitude == 0.0:
            return 0.0
        unit = vector / magnitude
        panel_normal = self._current_panel_normal()
        return float(_clamp(float(np.dot(unit, panel_normal)), 0.0, 1.0))

    def total_load_power(self) -> float:
        """Total power draw from all devices and motors."""
        regulated_load = sum(self._device_power(name) for name in self.device_states)
        battery_power_for_regulated = regulated_load / DC_DC_EFFICIENCY
        motor_power = self.motor_power_w * self.motor_count if self.motor_state else 0.0
        return battery_power_for_regulated + motor_power

    # -- Outputs --

    def battery_percentage(self) -> float:
        if self.battery_capacity_wh <= 0.0:
            return 0.0
        return 100.0 * self.battery_charge_wh / self.battery_capacity_wh

    def battery_voltage(self) -> float:
        return max(self.battery_voltage_v, 1e-3)

    def measured_battery_percentage(self) -> float:
        value = self.battery_percentage()
        return _clamp(value + random.gauss(0.0, self.battery_percentage_noise_std), 0.0, 100.0)

    def measured_battery_voltage(self) -> float:
        value = self.battery_voltage()
        min_v = BATTERY_VOLTAGE_CURVE[0][1]
        max_v = BATTERY_VOLTAGE_CURVE[-1][1]
        return _clamp(value + random.gauss(0.0, self.battery_voltage_noise_std), min_v, max_v)

    def measured_device_currents(self) -> Dict[str, float]:
        """Return measured device currents (A) with noise."""
        currents = {}
        for name in self.device_states:
            device_power_w = self._device_power(name)
            currents[name] = device_power_w / REGULATED_BUS_VOLTAGE
        return {
            name: max(0.0, value + random.gauss(0.0, self.device_current_noise_std))
            for name, value in currents.items()
        }

    def measured_motor_currents(self) -> Sequence[float]:
        """Return measured motor currents (A) with noise."""
        voltage = self.battery_voltage()
        base_current = (self.motor_power_w / voltage) if self.motor_state else 0.0
        return [
            max(0.0, base_current + random.gauss(0.0, self.device_current_noise_std))
            for _ in range(self.motor_count)
        ]

    def measured_solar_input_current(self) -> float:
        power = _clamp(
            self.solar_input_power + random.gauss(0.0, self.solar_input_noise_std),
            0.0, self.solar_panel_max_power,
        )
        return power / self.battery_voltage()

    def status(self) -> Dict[str, object]:
        """Return a snapshot of all measured outputs."""
        motor_currents = self.measured_motor_currents()
        device_currents = self.measured_device_currents()
        regulated_power = REGULATED_BUS_VOLTAGE * sum(device_currents.values())
        battery_voltage = self.battery_voltage()
        device_current_at_battery = (regulated_power / DC_DC_EFFICIENCY) / battery_voltage
        total_current_out = device_current_at_battery + sum(motor_currents)
        return {
            "net_power": self.solar_input_power - self.total_load_power(),
            "solar_input_current_measured": self.measured_solar_input_current(),
            "battery_percentage_measured": self.measured_battery_percentage(),
            "battery_voltage_measured": self.measured_battery_voltage(),
            "motor_currents_measured": motor_currents,
            "total_current_out_measured": total_current_out,
            "device_currents_measured": device_currents,
        }

    # -- Internal helpers --

    def _current_panel_normal(self) -> np.ndarray:
        base_normal = SOLAR_PANEL_NORMALS.get(
            self.solar_panel_state, SOLAR_PANEL_NORMALS["deployed"]
        )
        yaw_rad = math.radians(self.rover_yaw_deg)
        rotation = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        return rotation @ base_normal

    def _device_power(self, name: str) -> float:
        lo, hi = self.device_power_settings[name]
        if not self.device_states[name]:
            return lo
        elif self.device_health.get(name, DEVICE_HEALTH_NOMINAL) == DEVICE_HEALTH_FAULT:
            return hi + DEVICE_FAULT_EXTRA_POWER
        else:
            return hi

    def _update_battery_voltage(self) -> None:
        if self.battery_capacity_wh <= 0.0:
            self.battery_voltage_v = BATTERY_VOLTAGE_CURVE[0][1]
            return
        fraction = _clamp(self.battery_charge_wh / self.battery_capacity_wh, 0.0, 1.0)
        curve = BATTERY_VOLTAGE_CURVE
        for idx in range(1, len(curve)):
            p_hi, v_hi = curve[idx]
            if fraction <= p_hi:
                p_lo, v_lo = curve[idx - 1]
                if p_hi == p_lo:
                    self.battery_voltage_v = v_hi
                    return
                slope = (v_hi - v_lo) / (p_hi - p_lo)
                self.battery_voltage_v = v_lo + slope * (fraction - p_lo)
                return
        self.battery_voltage_v = curve[-1][1]
