from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PhysicsConfidence = Literal["authored", "estimated", "defaulted", "unknown"]
MobilityKind = Literal["static", "dynamic", "kinematic", "visual_only"]
CollisionKind = Literal["primitive", "mesh", "compound", "visual_only"]
PrimitiveKind = Literal["box", "cylinder", "sphere", "plane"]
ActionRepresentation = Literal["joint_position", "delta_ee_pose_gripper"]
VisualMeshRole = Literal["static_world", "dynamic_object"]
MaterialTier = Literal["preserved", "approximated", "unresolved"]


class PoseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: list[float]
    quaternion: list[float]

    @field_validator("position")
    @classmethod
    def position_has_three_values(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("position must contain three values")
        return value

    @field_validator("quaternion")
    @classmethod
    def quaternion_has_four_values(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("quaternion must contain four values")
        return value


class PhysicsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mass_kg: float | None = Field(default=None, gt=0)
    density: float | None = Field(default=None, gt=0)
    friction: list[float] = Field(default_factory=lambda: [0.8, 0.02, 0.001])
    restitution: float = Field(default=0.02, ge=0, le=1)
    confidence: PhysicsConfidence = "defaulted"

    @field_validator("friction")
    @classmethod
    def friction_has_three_nonnegative_values(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(item < 0 for item in value):
            raise ValueError("friction must contain three nonnegative values")
        return value


class CollisionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: CollisionKind
    primitive_type: PrimitiveKind | None = None
    size: list[float] | None = None
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    mesh_name: str | None = None
    mesh_path: str | None = None
    group: int = 2

    @field_validator("pos")
    @classmethod
    def pos_has_three_values(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("collision pos must contain three values")
        return value

    @field_validator("quat")
    @classmethod
    def quat_has_four_values(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("collision quat must contain four values")
        return value

    @model_validator(mode="after")
    def validate_collision_fields(self) -> "CollisionSpec":
        if self.kind == "primitive":
            if not self.primitive_type or not self.size:
                raise ValueError("primitive collision needs primitive_type and size")
        if self.kind == "mesh" and not self.mesh_name:
            raise ValueError("mesh collision needs mesh_name")
        return self


class VisualMaterialSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source_name: str | None = None
    rgba: list[float] = Field(default_factory=lambda: [0.6, 0.6, 0.6, 1.0])
    tier: MaterialTier = "approximated"
    preserved_features: list[str] = Field(default_factory=list)
    approximated_features: list[str] = Field(default_factory=list)
    unsupported_features: list[str] = Field(default_factory=list)
    critical: bool = False

    @field_validator("rgba")
    @classmethod
    def rgba_has_four_unit_values(cls, value: list[float]) -> list[float]:
        if len(value) != 4 or any(item < 0 or item > 1 for item in value):
            raise ValueError("rgba must contain four values between 0 and 1")
        return value


class VisualMeshSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh_name: str = Field(min_length=1)
    file: str = Field(min_length=1)
    material: str
    role: VisualMeshRole
    entity_id: str | None = None
    node_names: list[str] = Field(default_factory=list)
    group: int = 1


class SceneIRObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    name: str | None = None
    asset_id: str | None = None
    usd_prim: str | None = None
    source_visual_path: str | None = None
    visual_mesh: str | None = None
    visual_parts: list[VisualMeshSpec] = Field(default_factory=list)
    collision_meshes: list[str] = Field(default_factory=list)
    pose: PoseSpec
    dimensions: list[float]
    mobility: MobilityKind
    physics: PhysicsSpec
    collision: list[CollisionSpec] = Field(default_factory=list)
    support_id: str | None = None

    @field_validator("dimensions")
    @classmethod
    def dimensions_are_positive(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(item <= 0 for item in value):
            raise ValueError("dimensions must contain three positive values")
        return value


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    parent: str | None = None
    resolution: list[int] = Field(default_factory=lambda: [224, 224])
    fovy_deg: float = Field(default=58.0, gt=0, lt=180)

    @field_validator("resolution")
    @classmethod
    def resolution_has_two_positive_values(cls, value: list[int]) -> list[int]:
        if len(value) != 2 or any(item <= 0 for item in value):
            raise ValueError("resolution must contain two positive values")
        return value


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = "panda"
    name: str = "Franka Panda"
    mjcf_path: str
    base_position: list[float]
    base_yaw_rad: float
    arm_joint_names: list[str] = Field(default_factory=list)
    gripper_joint_names: list[str] = Field(default_factory=list)
    actuator_names: list[str] = Field(default_factory=list)
    home_qpos: list[float] = Field(default_factory=list)
    ee_site: str = "panda_gripper_site"
    gripper_max_width_m: float = Field(default=0.08, gt=0)

    @field_validator("base_position")
    @classmethod
    def base_position_has_three_values(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("base_position must contain three values")
        return value


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["pick_and_place"] = "pick_and_place"
    target_object: str
    destination_region: str = "generated_destination"
    destination_position: list[float]
    destination_size: list[float] = Field(default_factory=lambda: [0.12, 0.12, 0.08])
    support_id: str | None = None
    max_position_error_m: float = Field(default=0.05, gt=0)
    max_rotation_error_deg: float = Field(default=15.0, gt=0)
    stable_steps: int = Field(default=15, ge=0)
    max_pregrasp_target_drift_m: float = Field(default=0.03, gt=0)
    forbidden_contacts: list[str] = Field(default_factory=list)

    @field_validator("destination_position")
    @classmethod
    def destination_position_has_three_values(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("destination_position must contain three values")
        return value

    @field_validator("destination_size")
    @classmethod
    def destination_size_has_three_values(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(item <= 0 for item in value):
            raise ValueError("destination_size must contain three positive values")
        return value


class ResetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_xy_noise_m: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    object_yaw_noise_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    mass_scale: list[float] = Field(default_factory=lambda: [1.0, 1.0])
    friction_scale: list[float] = Field(default_factory=lambda: [1.0, 1.0])

    @field_validator("object_xy_noise_m", "object_yaw_noise_deg", "mass_scale", "friction_scale")
    @classmethod
    def ranges_have_two_values(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("randomization ranges must contain two values")
        return value


class PolicyContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_cameras: list[CameraSpec] = Field(default_factory=list)
    proprio: list[str] = Field(default_factory=lambda: ["joint_position", "joint_velocity", "gripper_width"])
    language_instruction: bool = True
    action_representation: ActionRepresentation = "delta_ee_pose_gripper"
    action_rate_hz: int = Field(default=10, gt=0)
    translation_bound_m: float = Field(default=0.03, gt=0)
    rotation_bound_rad: float = Field(default=0.15, gt=0)
    gripper_bounds: list[float] = Field(default_factory=lambda: [-1.0, 1.0])

    @field_validator("gripper_bounds")
    @classmethod
    def gripper_bounds_have_two_values(cls, value: list[float]) -> list[float]:
        if len(value) != 2 or value[0] >= value[1]:
            raise ValueError("gripper_bounds must be [min, max]")
        return value


class SceneIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    source_run_dir: str
    source_scene_glb: str
    source_scene_usd: str | None = None
    coordinate_system: dict[str, str] = Field(
        default_factory=lambda: {"units": "meters", "up_axis": "z", "handedness": "right"}
    )
    visual_scene: dict[str, object] = Field(default_factory=dict)
    visual_materials: list[VisualMaterialSpec] = Field(default_factory=list)
    static_visual_meshes: list[VisualMeshSpec] = Field(default_factory=list)
    bounds: list[float]
    objects: list[SceneIRObject]
    cameras: list[CameraSpec] = Field(default_factory=list)
    robot: RobotSpec
    task: TaskSpec
    reset: ResetSpec = Field(default_factory=ResetSpec)
    policy: PolicyContract = Field(default_factory=PolicyContract)

    @field_validator("bounds")
    @classmethod
    def bounds_are_positive(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(item <= 0 for item in value):
            raise ValueError("bounds must contain three positive values")
        return value

    def object_by_id(self, object_id: str) -> SceneIRObject:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        raise KeyError(object_id)


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode: int
    seed: int
    success: bool
    steps: int
    terminated_reason: str
    max_contact_force: float = 0.0
    collision_count: int = 0
    workspace_violation: bool = False
    object_drop: bool = False
    recovery_success: bool = False
    grasp_attempted: bool = False
    released_after_grasp: bool = False
    target_lifted: bool = False
    target_placed: bool = False
    final_target_distance_m: float | None = None
    trace_path: str | None = None
    video_path: str | None = None
    snapshot_path: str | None = None
    time_to_completion_s: float | None = None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    policy_id: str
    episodes: list[EpisodeResult] = Field(default_factory=list)
    success_rate: float = 0.0
    collision_rate: float = 0.0
    object_drop_rate: float = 0.0
    workspace_violation_rate: float = 0.0
    max_contact_force: float = 0.0
    compile_artifacts: dict[str, str] = Field(default_factory=dict)
    visualization_artifacts: list[str] = Field(default_factory=list)
    task_feasibility: dict[str, object] = Field(default_factory=dict)
    visual_twin: dict[str, object] = Field(default_factory=dict)
    physics_settling: dict[str, object] = Field(default_factory=dict)
    import_success: bool = False
    task_feasibility_success: bool = False
    policy_success: bool = False
    outcome: dict[str, object] = Field(default_factory=dict)


class CompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_ir_path: str
    xml_path: str
    mjb_path: str | None = None
    mesh_dir: str
    compile_report_path: str
    object_count: int
    dynamic_object_count: int
    visual_only_object_count: int
