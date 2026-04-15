#!/usr/bin/env python3
"""Profile each stage of the 40m environment loading pipeline (CPU-only, no USD).

Usage:
    cd /home/sim2real1/new_lunar_sim && python3 tools/profile_loading.py
"""

import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def timed(label):
    """Context manager that prints elapsed time."""
    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.t0
            print(f"  {label:.<50s} {self.elapsed:7.3f}s")
    return Timer()


def profile_terrain_mesh():
    """Profile TerrainMeshBuilder._build_grid for 40m and 100m."""
    from terrain.mesh.terrain_mesh import TerrainMeshBuilder

    print("\n=== TerrainMeshBuilder._build_grid ===")

    for size_m, res in [(40, 0.02), (100, 0.02)]:
        px = int(size_m / res)
        with timed(f"{size_m}m ({px}x{px} = {px*px/1e6:.1f}M verts)"):
            builder = TerrainMeshBuilder(
                sim_width_px=px, sim_length_px=px,
                grid_size=res, root_path="/test",
                pxr_utils=None,
            )
        print(f"    vertices: {builder.vertices.shape}, "
              f"indices: {builder._indices.shape}, "
              f"uvs: {builder._uvs.shape}")


def profile_base_terrain():
    """Profile base terrain generation."""
    from terrain.config import BaseTerrainGeneratorConf
    from terrain.procedural.base_terrain import BaseTerrainGenerator

    print("\n=== BaseTerrainGenerator ===")

    for size_m in [40, 100]:
        cfg = BaseTerrainGeneratorConf(x_size=size_m, y_size=size_m, resolution=0.02, seed=42)
        gen = BaseTerrainGenerator(cfg)
        with timed(f"{size_m}m generate (is_yard=True)"):
            dem = gen.generate(is_yard=True)
        print(f"    DEM shape: {dem.shape}, range: [{dem.min():.3f}, {dem.max():.3f}]")


def profile_crater_distribution():
    """Profile crater distribution (Poisson + rejection)."""
    from terrain.config import CraterDistributionConf
    from terrain.procedural.crater_distribution import CraterDistributor

    print("\n=== CraterDistributor ===")

    for size_m in [40, 100]:
        cfg = CraterDistributionConf(
            x_size=size_m, y_size=size_m,
            densities=[0.01, 0.02, 0.03],
            radius=[[1.5, 2.5], [0.75, 1.5], [0.25, 0.5]],
            num_repeat=0, seed=42,
        )
        dist = CraterDistributor(cfg)
        with timed(f"{size_m}m distribution"):
            coords, radii = dist.run()
        print(f"    craters: {len(radii)}")


def profile_crater_generation():
    """Profile crater stamping onto DEM."""
    from terrain.config import (
        CraterGeneratorConf, CraterDistributionConf, RealisticCraterConf,
    )
    from terrain.procedural.crater_distribution import CraterDistributor
    from terrain.procedural.realistic_crater_generator import RealisticCraterGenerator

    print("\n=== RealisticCraterGenerator (stamp onto DEM) ===")

    size_m = 40
    res = 0.02
    px = int(size_m / res)

    dem = np.random.default_rng(42).uniform(-0.3, 0.3, (px, px)).astype(np.float32)

    dist_cfg = CraterDistributionConf(
        x_size=size_m, y_size=size_m,
        densities=[0.01, 0.02, 0.03],
        radius=[[1.5, 2.5], [0.75, 1.5], [0.25, 0.5]],
        seed=42,
    )
    coords, radii = CraterDistributor(dist_cfg).run()

    gen_cfg = CraterGeneratorConf(
        profiles_path="assets/Terrains/crater_spline_profiles.pkl",
        crater_mode="realistic", resolution=res, z_scale=0.4, seed=42,
    )
    rc_cfg = RealisticCraterConf()
    gen = RealisticCraterGenerator(gen_cfg, rc_cfg)

    with timed(f"stamp {len(radii)} craters onto {px}x{px} DEM"):
        dem_out, mask, cdata = gen.generate_craters(dem, coords, radii)
    print(f"    output DEM: {dem_out.shape}")


def profile_rock_distribution():
    """Profile rock placement + overlap rejection."""
    from objects.rock_distribution import RockDistributor, RockPlacement, CraterData

    print("\n=== RockDistributor ===")

    rng = np.random.default_rng(42)
    tw, th = 40.0, 40.0

    # background_scatter (most common)
    with timed(f"background_scatter density=0.2"):
        pl = RockDistributor.background_scatter(rng, tw, th, 0.02, density=0.2,
                                                 d_min=0.2, d_max=1.0, alpha=3.0)
    print(f"    rocks generated: {len(pl.diameters)}")

    with timed(f"remove_overlaps ({len(pl.diameters)} rocks)"):
        pl2 = RockDistributor.remove_overlaps(pl, min_gap=0.01, native_rock_size=0.3)
    print(f"    rocks kept: {len(pl2.diameters)}")

    # Larger set
    with timed(f"background_scatter density=1.0"):
        pl3 = RockDistributor.background_scatter(rng, tw, th, 0.02, density=1.0,
                                                  d_min=0.1, d_max=0.5, alpha=2.5)
    print(f"    rocks generated: {len(pl3.diameters)}")

    with timed(f"remove_overlaps ({len(pl3.diameters)} rocks)"):
        pl4 = RockDistributor.remove_overlaps(pl3, min_gap=0.01, native_rock_size=0.3)
    print(f"    rocks kept: {len(pl4.diameters)}")


def profile_full_moonyard():
    """Profile the full MoonyardGenerator pipeline."""
    from terrain.config import MoonYardConf, BaseTerrainGeneratorConf, CraterDistributionConf, CraterGeneratorConf, RealisticCraterConf
    from terrain.procedural.moonyard_generator import MoonyardGenerator

    print("\n=== Full MoonyardGenerator Pipeline (40m) ===")

    cfg = MoonYardConf(
        is_yard=True,
        base_terrain_generator=BaseTerrainGeneratorConf(
            x_size=40, y_size=40, resolution=0.02, seed=42,
        ),
        crater_distribution=CraterDistributionConf(
            x_size=40, y_size=40,
            densities=[0.01, 0.02, 0.03],
            radius=[[1.5, 2.5], [0.75, 1.5], [0.25, 0.5]],
            seed=42,
        ),
        crater_generator=CraterGeneratorConf(
            profiles_path="assets/Terrains/crater_spline_profiles.pkl",
            crater_mode="realistic", resolution=0.02, z_scale=0.4, seed=42,
        ),
        realistic_crater=RealisticCraterConf(),
    )

    gen = MoonyardGenerator(cfg)

    with timed("MoonyardGenerator.randomize() total"):
        dem, mask, craters = gen.randomize()
    print(f"    DEM: {dem.shape}, craters: {len(craters)}")


def profile_update_heights():
    """Profile DEM -> vertex height update."""
    from terrain.mesh.terrain_mesh import TerrainMeshBuilder

    print("\n=== update_heights ===")

    for size_m in [40, 100]:
        px = int(size_m / 0.02)
        builder = TerrainMeshBuilder(
            sim_width_px=px, sim_length_px=px,
            grid_size=0.02, root_path="/test", pxr_utils=None,
        )
        dem = np.random.default_rng(42).uniform(-1, 1, (px, px)).astype(np.float32)

        with timed(f"{size_m}m update_heights ({px}x{px})"):
            builder.update_heights(dem)


if __name__ == "__main__":
    print("=" * 60)
    print("  Loading Pipeline Profiler")
    print("=" * 60)

    t_total = time.perf_counter()

    profile_terrain_mesh()
    profile_base_terrain()
    profile_crater_distribution()
    profile_crater_generation()
    profile_rock_distribution()
    profile_full_moonyard()
    profile_update_heights()

    elapsed = time.perf_counter() - t_total
    print(f"\n{'TOTAL':.<50s} {elapsed:7.3f}s")
