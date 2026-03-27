from __future__ import annotations

import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass
class PhysicsConf:
    dt: float = 0.016666
    gravity: Tuple[float, float, float] = (0.0, 0.0, -1.62)
    enable_ccd: bool = True
    solver_type: str = "PGS"
    broadphase_type: str = "SAP"
    substeps: Optional[int] = None
    use_gpu_pipeline: Optional[bool] = None
    worker_thread_count: Optional[int] = None
    use_fabric: Optional[bool] = None
    enable_scene_query_support: Optional[bool] = None
    gpu_max_rigid_contact_count: Optional[int] = None
    gpu_max_rigid_patch_contact_count: Optional[int] = None
    gpu_found_lost_pairs_capacity: Optional[int] = None
    gpu_total_aggregate_pairs_capacity: Optional[int] = None
    gpu_max_soft_body_contacts: Optional[int] = None
    gpu_max_particle_contacts: Optional[int] = None
    gpu_heap_capacity: Optional[int] = None
    gpu_temp_buffer_capacity: Optional[int] = None
    gpu_max_num_partitions: Optional[int] = None
    gpu_collision_stack_size: Optional[int] = None
    enable_stabilization: Optional[bool] = None
    bounce_threshold_velocity: Optional[float] = None
    friction_offset_threshold: Optional[float] = None
    friction_correlation_distance: Optional[float] = None

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.solver_type not in ("PGS", "TGS"):
            raise ValueError(
                f"solver_type must be 'PGS' or 'TGS', got '{self.solver_type}'"
            )
        if self.broadphase_type not in ("SAP", "MBP", "GPU"):
            raise ValueError(
                f"broadphase_type must be 'SAP', 'MBP', or 'GPU', got '{self.broadphase_type}'"
            )
        # Build physics_scene_args dict with only non-None values
        self.physics_scene_args = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                self.physics_scene_args[field.name] = value


@dataclasses.dataclass
class RendererConf:
    renderer: str = "RayTracedLighting"
    headless: bool = False
    width: int = 1280
    height: int = 720
    samples_per_pixel_per_frame: int = 32
    max_bounces: int = 6
    subdiv_refinement_level: int = 0

    def __post_init__(self):
        if self.renderer not in ("RayTracedLighting", "PathTracing"):
            raise ValueError(
                f"renderer must be 'RayTracedLighting' or 'PathTracing', got '{self.renderer}'"
            )
