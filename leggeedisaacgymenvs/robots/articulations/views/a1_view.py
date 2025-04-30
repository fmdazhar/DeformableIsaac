

from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class A1View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "A1View",
        track_contact_forces=False,
        prepare_contact_sensors=True,
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)
        self._thigh = RigidPrimView(
            prim_paths_expr="/World/envs/.*/a1/.*_thigh",
            name="thigh_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self._calf = RigidPrimView(
            prim_paths_expr="/World/envs/.*/a1/.*_calf",
            name="calf_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self._base = RigidPrimView(
            prim_paths_expr="/World/envs/.*/a1/trunk",
            name="base_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self._foot = RigidPrimView(
            prim_paths_expr="/World/envs/.*/a1/.*_foot",
            name="foot_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )

        self._foot_material_path = "/World/envs/.*/a1/PhysicsMaterial*"

    def get_thigh_transforms(self):
        return self._thigh.get_world_poses()

    def is_thigh_below_threshold(self, threshold, ground_heights=None):
        thigh_pos, _ = self._thigh.get_world_poses()
        thigh_heights = thigh_pos.view((-1, 4, 3))[:, :, 2]
        if ground_heights is not None:
            thigh_heights -= ground_heights
        return (
            (thigh_heights[:, 0] < threshold)
            | (thigh_heights[:, 1] < threshold)
            | (thigh_heights[:, 2] < threshold)
            | (thigh_heights[:, 3] < threshold)
        )

    def is_base_below_threshold(self, threshold, ground_heights):
        base_pos, _ = self.get_world_poses()
        base_heights = base_pos[:, 2]
        base_heights -= ground_heights
        return base_heights[:] < threshold
