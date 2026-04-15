# sim2moon

A lunar surface rover simulation framework built on NVIDIA Isaac Sim.
Provides procedural terrain generation, physics-based wheel-terrain interaction (terramechanics), multi-robot support, ROS2 integration, and sensor simulation.

## Features

| Feature | Description |
|---------|-------------|
| **Procedural Terrain** | Perlin noise base terrain + Poisson-distributed craters + realistic deformation |
| **DEM-based Landscape** | Real lunar South Pole DEM data for background terrain |
| **Terramechanics** | Bekker-Janosi wheel-terrain interaction with slip-sinkage coupling |
| **Terrain Deformation** | Real-time heightmap deformation from wheel contact |
| **Multi-Robot** | Concurrent multi-robot operation via ROS2 domain IDs |
| **Sensor Simulation** | Camera, IMU, LiDAR |
| **ROS2 Integration** | TF, services (spawn/reset/teleport), topic publishing |
| **Celestial Simulation** | JPL Ephemeris-based sun/earth positioning |
| **Visual Effects** | Starfield (4M stars), Earthshine, dust particles |
| **Synthetic Data Generation** | Scene randomization + automatic labeling |
| **UDP Protocol** | S2R_ICD rover state packet transmission |

## Project Structure

```
sim2moon/
├── main.py                     # Entry point (Hydra config loading)
├── run_ros2.sh                 # Launch script with ROS2 environment setup
├── config/                     # Hydra YAML configuration
│   ├── config.yaml             # Default selection (environment/rendering/physics)
│   ├── environment/            # Environment presets (14+)
│   ├── rendering/              # Rendering quality settings
│   ├── physics/                # Physics engine settings
│   └── mode/                   # Execution mode (ROS2, SDG)
├── core/                       # Simulation core
│   ├── simulation_manager.py   # Simulation lifecycle management
│   ├── config_factory.py       # Config instantiation and validation
│   └── pxr_utils.py            # USD/PyXR utilities
├── environments/               # Environment definitions
│   ├── lunar_yard.py           # LunarYard environment (terrain + physics + effects)
│   └── base_environment.py     # Abstract base class
├── terrain/                    # Terrain system
│   ├── terrain_manager.py      # Unified terrain interface
│   ├── landscape_builder.py    # DEM-based background landscape
│   ├── procedural/             # Procedural generation (craters, noise)
│   ├── deformation/            # Real-time terrain deformation
│   ├── mesh/                   # USD mesh builder
│   └── materials/              # MDL shaders
├── robots/                     # Robot control
│   ├── robot_manager.py        # Multi-robot spawn/reset
│   ├── robot.py                # Robot instance (drive, sensors)
│   └── subsystems/             # Power/thermal/radio models
├── sensors/                    # Sensors
│   ├── camera.py / camera_ros2.py
│   ├── imu.py
│   └── lidar.py
├── bridges/                    # Communication bridges
│   ├── ros2_bridge.py          # ROS2 integration (TF, services, topics)
│   └── udp_bridge.py           # S2R_ICD UDP protocol
├── physics/                    # Physics models
│   └── terramechanics.py       # Bekker-Janosi wheel-terrain dynamics
├── rendering/                  # Rendering
│   ├── renderer.py             # RTX ray tracing configuration
│   └── post_processing.py      # Lens flare, motion blur
├── effects/                    # Visual effects
│   ├── starfield.py            # Procedural star background
│   ├── earthshine.py           # Earth reflected light
│   └── dust_manager.py         # Dust particles
├── celestial/                  # Celestial simulation
│   └── stellar_engine.py       # JPL DE421-based celestial positioning
├── objects/                    # Rock distribution and placement
├── data_generation/            # Synthetic data generation
├── tools/                      # Utility scripts
├── tests/                      # Tests (23 modules)
├── assets/                     # USD models, DEM data, textures
└── docs/                       # Technical documentation
```

## Requirements

### System Requirements

- **OS**: Ubuntu 22.04+
- **GPU**: NVIDIA RTX GPU (RTX 3070 or higher recommended)
- **NVIDIA Driver**: 535.129.03+
- **Python**: 3.10+

### Software

- **NVIDIA Isaac Sim** (5.0+)
- **ROS2 Humble** (required for ROS2 mode)

## Installation

### 1. Install Isaac Sim

Follow the official [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) installation guide.
The default installation path is `$HOME/isaacsim`.

### 2. Clone the Repository

```bash
git clone https://github.com/eugene0429/sim2moon.git
cd sim2moon
```

### 3. Install Python Dependencies

```bash
# Install within Isaac Sim's Python environment
$HOME/isaacsim/python.sh -m pip install hydra-core omegaconf numpy

# For celestial simulation
$HOME/isaacsim/python.sh -m pip install skyfield scipy

# Development tools (optional)
$HOME/isaacsim/python.sh -m pip install pytest
```

### 4. Verify Assets

Ensure the following items exist in the `assets/` directory:

```
assets/
├── Terrains/           # Crater profiles, DEM data
├── USD_Assets/         # Robot and rock USD models
├── Textures/           # Earth texture, etc.
└── Ephemeris/          # JPL DE421 ephemeris data (de421.bsp)
```

## Usage

### Basic Launch

```bash
# Default environment (lunar_yard_40m, ray tracing)
$HOME/isaacsim/python.sh main.py
```

### ROS2 Mode

```bash
# Automatically sets ROS2 environment variables and launches
./run_ros2.sh

# Specify a particular environment
./run_ros2.sh environment=lunar_yard_40m_workshop_full_husky
```

### Configuration Overrides

Settings can be overridden from the command line using Hydra:

```bash
# Use a different environment preset
$HOME/isaacsim/python.sh main.py environment=lunar_yard_20m_playground

# Change rendering mode
$HOME/isaacsim/python.sh main.py rendering=path_tracing

# Headless mode
$HOME/isaacsim/python.sh main.py rendering.renderer.headless=true

# Change seed
$HOME/isaacsim/python.sh main.py environment.seed=123
```

### Pre-configured Environment Presets

| Environment | Size | Description |
|-------------|------|-------------|
| `lunar_yard_20m` | 20m | Small-scale testing |
| `lunar_yard_20m_playground` | 20m | Tilted terrain + terramechanics |
| `lunar_yard_20m_deformable` | 20m | Terrain deformation enabled |
| `lunar_yard_40m` | 40m | Default environment |
| `lunar_yard_40m_realistic_rocks` | 40m | Realistic rock distribution |
| `lunar_yard_40m_deformable` | 40m | Terrain deformation enabled |
| `lunar_yard_80m` | 80m | Large-scale environment |
| `lunar_yard_100m_workshop_full` | 100m | Full workshop configuration |

## Utility Tools

```bash
# Keyboard teleoperation
$HOME/isaacsim/python.sh tools/teleop_keyboard.py

# Spawn rover
$HOME/isaacsim/python.sh tools/spawn_rover.py

# Camera pan control
$HOME/isaacsim/python.sh tools/pan_camera.py

# Stereo camera viewer (ROS2)
python tools/stereo_viewer.py --left /robot/left/image_raw --right /robot/right/image_raw

# UDP packet receiver test
python tools/udp_receiver.py
```

## Testing

```bash
$HOME/isaacsim/python.sh -m pytest tests/ -v
```

## Configuration System

Uses a hierarchical YAML configuration system based on Hydra + OmegaConf.

```
config/
├── config.yaml             # Default selection
├── environment/            # Per-environment full configuration
├── rendering/              # ray_tracing.yaml, path_tracing.yaml
├── physics/                # Physics engine parameters
└── mode/                   # ROS2, SDG modes
```

Example configuration (`config/environment/lunar_yard_40m.yaml`):

```yaml
name: LunarYard
seed: 42
physics_dt: 0.0333
rendering_dt: 0.0333
mode: ROS2

lunaryard_settings:
  lab_length: 40.0
  lab_width: 40.0
  resolution: 0.02

terrain_manager:
  moon_yard:
    crater_generator:
      z_scale: 0.2
    crater_distribution:
      densities: [0.025, 0.05, 0.5]
    base_terrain_generator:
      max_elevation: 0.5

stellar_engine_settings:
  start_date: { year: 2024, month: 5, day: 1 }
  time_scale: 36000

rocks_settings:
  enable: true
```

## Documentation

- [Terramechanics (Bekker-Janosi Model)](docs/terramechanics.md)
- [Realistic Crater Generation](docs/realistic_craters.md)
- [Celestial Effects (Starfield & Earthshine)](docs/celestial_effects.md)
- [Terrain Materials](docs/terrain_materials.md)
- [Lunar Reflection Analysis](docs/lunar_reflection_analysis.md)

## License

This project is developed for research and educational purposes.
