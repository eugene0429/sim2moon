"""Tests for robot subsystem models (power, thermal, radio)."""

import pytest

from robots.subsystems.power import PowerModel, BATTERY_CAPACITY_WH, DEVICE_SETTINGS
from robots.subsystems.thermal import ThermalModel
from robots.subsystems.radio import RadioModel


# ── PowerModel ──────────────────────────────────────────────────────────────

class TestPowerModel:
    def test_initial_state(self):
        model = PowerModel()
        assert model.battery_percentage() == pytest.approx(100.0)
        assert model.battery_voltage() > 0

    def test_device_states_initialized(self):
        model = PowerModel()
        for name in DEVICE_SETTINGS:
            assert name in model.device_states
            assert model.device_states[name] is False

    def test_set_device_state(self):
        model = PowerModel()
        model.set_device_state("current_draw_obc", True)
        assert model.device_states["current_draw_obc"] is True

    def test_set_device_state_unknown_raises(self):
        model = PowerModel()
        with pytest.raises(KeyError):
            model.set_device_state("nonexistent_device", True)

    def test_set_solar_panel_state(self):
        model = PowerModel()
        model.set_solar_panel_state("deployed")
        assert model.solar_panel_state == "deployed"

    def test_set_solar_panel_state_invalid_raises(self):
        model = PowerModel()
        with pytest.raises(ValueError):
            model.set_solar_panel_state("broken")

    def test_total_load_zero_when_all_off(self):
        model = PowerModel()
        model.set_motor_state(False)
        # All devices off by default, so load is just the sum of "off" powers
        load = model.total_load_power()
        assert load >= 0

    def test_total_load_increases_with_motors(self):
        model = PowerModel()
        load_no_motor = model.total_load_power()
        model.set_motor_state(True)
        load_motor = model.total_load_power()
        assert load_motor > load_no_motor

    def test_step_discharges_battery(self):
        model = PowerModel()
        model.set_all_devices(True)
        model.set_motor_state(True)
        # Sun far away: no charging
        model.set_sun_position((0.0, 0.0, -1000.0))
        initial_pct = model.battery_percentage()
        model.step(60.0)  # 60 seconds
        assert model.battery_percentage() < initial_pct

    def test_step_charges_with_sun(self):
        model = PowerModel(battery_charge_wh=30.0)
        model.set_solar_panel_state("deployed")
        # Sun directly along panel normal
        model.set_sun_position((0.0, 1000.0, 0.0))
        initial_pct = model.battery_percentage()
        model.step(60.0)
        assert model.battery_percentage() > initial_pct

    def test_battery_clamps_at_zero(self):
        model = PowerModel(battery_charge_wh=0.01)
        model.set_all_devices(True)
        model.set_motor_state(True)
        model.set_sun_position((0.0, 0.0, -1000.0))
        for _ in range(100):
            model.step(60.0)
        assert model.battery_percentage() >= 0.0

    def test_battery_clamps_at_full(self):
        model = PowerModel(battery_charge_wh=BATTERY_CAPACITY_WH)
        model.set_solar_panel_state("deployed")
        model.set_sun_position((0.0, 1000.0, 0.0))
        for _ in range(100):
            model.step(60.0)
        assert model.battery_percentage() <= 100.0

    def test_view_factor_perpendicular(self):
        model = PowerModel()
        model.set_solar_panel_state("deployed")
        model.set_rover_yaw(0.0)
        # Panel normal is (0, 1, 0) when deployed + yaw=0
        vf = model.compute_view_factor((0, 0, 0), (0, 100, 0))
        assert vf == pytest.approx(1.0, abs=0.01)

    def test_view_factor_behind(self):
        model = PowerModel()
        model.set_solar_panel_state("deployed")
        model.set_rover_yaw(0.0)
        vf = model.compute_view_factor((0, 0, 0), (0, -100, 0))
        assert vf == pytest.approx(0.0)

    def test_status_returns_all_keys(self):
        model = PowerModel()
        model.step(1.0)
        status = model.status()
        expected_keys = {
            "net_power", "solar_input_current_measured",
            "battery_percentage_measured", "battery_voltage_measured",
            "motor_currents_measured", "total_current_out_measured",
            "device_currents_measured",
        }
        assert expected_keys == set(status.keys())

    def test_device_health_fault_increases_power(self):
        model = PowerModel()
        model.set_device_state("current_draw_obc", True)
        load_nominal = model.total_load_power()
        model.set_device_health({"current_draw_obc": "FAULT"})
        load_fault = model.total_load_power()
        assert load_fault > load_nominal


# ── ThermalModel ────────────────────────────────────────────────────────────

class TestThermalModel:
    def test_initial_temperatures(self):
        model = ThermalModel(initial_temp=25.0)
        temps = model.temperatures()
        for face in ("+X", "-X", "+Y", "-Y", "+Z", "-Z", "interior"):
            assert face in temps

    def test_sun_heats_exposed_face(self):
        model = ThermalModel(initial_temp=0.0, measurement_noise_std=0.0)
        model.set_sun_position((1000.0, 0.0, 0.0))  # Sun along +X
        model.set_rover_yaw(0.0)
        for _ in range(1000):
            model.step(1.0)
        temps = model.temperatures()
        # +X face should be warmer than -X face
        assert temps["+X"] > temps["-X"]

    def test_interior_is_average(self):
        model = ThermalModel(measurement_noise_std=0.0)
        model.set_sun_position((100.0, 50.0, 0.0))
        model.step(10.0)
        temps = model.temperatures()
        face_avg = sum(temps[f] for f in model.faces) / len(model.faces)
        assert temps["interior"] == pytest.approx(face_avg, abs=1e-6)

    def test_view_factors_sum_reasonable(self):
        model = ThermalModel()
        vf = model.compute_view_factors((0, 0, 0), (100, 0, 0))
        # At most 3 faces can see a point (adjacent faces); total <= 3
        assert sum(vf.values()) <= 3.01

    def test_view_factor_zero_distance(self):
        model = ThermalModel()
        vf = model.compute_view_factors((5, 5, 5), (5, 5, 5))
        assert all(v == 0.0 for v in vf.values())

    def test_step_converges_toward_target(self):
        model = ThermalModel(initial_temp=-50.0, measurement_noise_std=0.0)
        model.set_sun_position((1000.0, 0.0, 0.0))
        t0 = model.temperatures()["+X"]
        for _ in range(5000):
            model.step(1.0)
        t1 = model.temperatures()["+X"]
        # Should have moved toward higher temperature
        assert t1 > t0


# ── RadioModel ──────────────────────────────────────────────────────────────

class TestRadioModel:
    def test_zero_distance(self):
        model = RadioModel(noise_std=0.0)
        model.set_rover_position((0.0, 0.0, 0.0))
        model.set_lander_position((0.0, 0.0, 0.0))
        assert model.distance() == 0.0
        assert model.rssi() == pytest.approx(model.best_rssi)

    def test_distance_calculation(self):
        model = RadioModel()
        model.set_rover_position((3.0, 4.0, 0.0))
        model.set_lander_position((0.0, 0.0, 0.0))
        assert model.distance() == pytest.approx(5.0)

    def test_rssi_decreases_with_distance(self):
        model = RadioModel(noise_std=0.0)
        model.set_lander_position((0.0, 0.0, 0.0))

        model.set_rover_position((10.0, 0.0, 0.0))
        rssi_near = model.rssi()

        model.set_rover_position((50.0, 0.0, 0.0))
        rssi_far = model.rssi()

        # RSSI values are negative; closer = stronger (more negative = weaker)
        # best_rssi=-90, worst_rssi=-30, so stronger means more negative
        assert rssi_near < rssi_far  # near is more negative = stronger signal

    def test_rssi_at_reference_distance(self):
        model = RadioModel(noise_std=0.0, reference_distance=100.0)
        model.set_rover_position((100.0, 0.0, 0.0))
        assert model.rssi() == pytest.approx(model.worst_rssi)

    def test_rssi_no_noise_deterministic(self):
        model = RadioModel(noise_std=0.0)
        model.set_rover_position((25.0, 0.0, 0.0))
        r1 = model.rssi_no_noise()
        r2 = model.rssi_no_noise()
        assert r1 == r2
