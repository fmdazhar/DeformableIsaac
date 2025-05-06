    # def update_command_curriculum(self, env_ids):
    #     """ Implements a curriculum of increasing commands

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     # If the tracking reward is above 80% of the maximum, increase the range of commands
    #     if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length_s > 0.8 * self.reward_scales["tracking_lin_vel"]:
    #         self.command_x_range[0] = np.clip(self.command_x_range[0] - 0.2, -self.limit_vel_x[0], 0.).item()
    #         self.command_x_range[1] = np.clip(self.command_x_range[1] + 0.2, 0., self.limit_vel_x[1]).item()

    #         # Increase the range of commands for y
    #         self.command_y_range[0] = np.clip(self.command_y_range[0] - 0.2, -self.limit_vel_y[0], 0.).item()
    #         self.command_y_range[1] = np.clip(self.command_y_range[1] + 0.2, 0., self.limit_vel_y[1]).item()
        
    #     if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length_s > 0.8 * self.reward_scales["tracking_ang_vel"]:
    #     # Increase the range of commands for yaw
    #         self.command_yaw_range[0] = np.clip(self.command_yaw_range[0] - 0.2, -self.limit_vel_yaw[0], 0.).item()
    #         self.command_yaw_range[1] = np.clip(self.command_yaw_range[1] + 0.2, 0., self.limit_vel_yaw[1]).item()
    


    def _visualize_terrain_heights(self):
        """
        Spawns (or updates) a PointInstancer of small spheres for every cell in self.height_samples,
        but only for the main terrain region (excluding the border).
        """
        stage = self._stage  # or get_current_stage()

        # 1) Create a dedicated Scope for the debug instancer
        parent_scope_path = "/World/DebugTerrainHeights"
        parent_scope_prim = stage.GetPrimAtPath(parent_scope_path)
        if not parent_scope_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=parent_scope_path,
                attributes={}
            )

        # 2) Construct a PointInstancer prim if not already there
        point_instancer_path = f"{parent_scope_path}/terrain_points_instancer"
        point_instancer_prim = stage.GetPrimAtPath(point_instancer_path)
        if not point_instancer_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=point_instancer_path,
                attributes={}
            )

        point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(point_instancer_path))

        # 3) Create/ensure we have a single prototype (Sphere) under the PointInstancer
        prototype_index = 0
        proto_path = f"{point_instancer_path}/prototype_Sphere"
        prototype_prim = stage.GetPrimAtPath(proto_path)
        if not prototype_prim.IsValid():
            # Create a sphere prototype
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=proto_path,
                attributes={"radius": 0.02},  # adjust sphere size as you wish
            )
        # This step ensures the point-instancer references the prototype as well
        if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
            point_instancer.GetPrototypesRel().AddTarget(proto_path)

        # 4) Build up the positions (and protoIndices) for each cell of the *main* height field
        tot_rows = self.terrain.tot_rows   # i dimension
        tot_cols = self.terrain.tot_cols   # j dimension
        border   = self.terrain.border     # integer # of “cells” that define the border thickness

        positions = []
        proto_indices = []

        # Only iterate within the interior region [border, (tot_rows - border)) and [border, (tot_cols - border))
        for i in range(border, tot_rows - border):
            for j in range(border, tot_cols - border):
                # Convert row/col -> world coordinates
                px = i * self.terrain.horizontal_scale - self.terrain.border_size
                py = j * self.terrain.horizontal_scale - self.terrain.border_size
                pz = float(self.height_samples[i, j] * self.terrain.vertical_scale)

                positions.append(Gf.Vec3f(px, py, pz))
                proto_indices.append(prototype_index)

        positions_array = Vt.Vec3fArray(positions)
        proto_indices_array = Vt.IntArray(proto_indices)

        # 5) Assign the arrays to the PointInstancer
        point_instancer.CreatePositionsAttr().Set(positions_array)
        point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

        # Optionally give these debug spheres a color by modifying the prototype itself:
        sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(proto_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 1.0)])


    def _visualize_depression_indices(self):
        """
        Creates (or updates) a PointInstancer of small spheres at z=0 
        for each (x,y) entry
        """

        stage = self._stage  # Or get_current_stage()

        # 1) Create a dedicated Scope for debugging these indices
        debug_scope_path = "/World/DebugDepressionIndices"
        if not stage.GetPrimAtPath(debug_scope_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=debug_scope_path,
                attributes={}
            )

        # 2) Create a PointInstancer for all depression indices
        instancer_path = f"{debug_scope_path}/DepressionIndicesPointInstancer"
        if not stage.GetPrimAtPath(instancer_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=instancer_path,
                attributes={}
            )
        point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_path))

        # 3) Make sure there's a prototype sphere
        sphere_proto_path = f"{instancer_path}/DepressionIndexSphere"
        if not stage.GetPrimAtPath(sphere_proto_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=sphere_proto_path,
                attributes={"radius": 0.02},  # adjust size as desired
            )
        # Ensure the instancer references the prototype
        if len(point_instancer.GetPrototypesRel().GetTargets()) == 0:
            point_instancer.GetPrototypesRel().AddTarget(sphere_proto_path)

        # 4) Collect positions and prototype indices
        positions = []
        proto_indices = []

        prototype_index = 0  # single prototype

        for terrain_entry in self.terrain_details:
            terrain_name = terrain_entry[15]
            if terrain_name == "central_depression_terrain":
                bx_start = int(terrain_entry[10])
                bx_end   = int(terrain_entry[11])
                by_start = int(terrain_entry[12])
                by_end   = int(terrain_entry[13])

                # For each (i, j) in that rectangle
                for i in range(bx_start, bx_end):
                    for j in range(by_start, by_end):
                        # Convert heightfield indices to world coordinates
                        px = i * self.terrain.horizontal_scale - self.terrain.border_size
                        py = j * self.terrain.horizontal_scale - self.terrain.border_size
                        pz = 0.0  # place at ground (z=0)
                        positions.append(Gf.Vec3f(px, py, pz))
                        proto_indices.append(prototype_index)

        positions_array = Vt.Vec3fArray(positions)
        proto_indices_array = Vt.IntArray(proto_indices)

        # 5) Assign to the PointInstancer
        point_instancer.CreatePositionsAttr().Set(positions_array)
        point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

        # (Optional) Color the debug spheres differently
        sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(sphere_proto_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])  # green for clarity



















            # Iterate by index instead of zip
            for idx in range(px.shape[0]):
                i = int(px[idx].item())
                j = int(py[idx].item())
                # Convert (i,j) indices into world coordinates.
                # Here we assume that each cell spans a distance equal to terrain.horizontal_scale,
                # and that the terrain's origin offset is given by terrain.border_size.
                cell_x_min = (i-0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_x_max = (i + 0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y_min = (j-0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y_max = (j + 0.5) * self.terrain.horizontal_scale - self.terrain.border_size
                cell_x = i * self.terrain.horizontal_scale - self.terrain.border_size
                cell_y = j * self.terrain.horizontal_scale - self.terrain.border_size

                # For each cell defined by cell_x_min, cell_x_max, cell_y_min, cell_y_max:
                mask = (
                    (particle_positions_np[:, 0] >= cell_x_min) &
                    (particle_positions_np[:, 0] < cell_x_max) &
                    (particle_positions_np[:, 1] >= cell_y_min) &
                    (particle_positions_np[:, 1] < cell_y_max)
                )
                if np.any(mask):
                    top_z = float(np.max(particle_positions_np[mask, 2]))
                    top_z = min(top_z, 1*self.terrain.vertical_scale)
                    self.height_samples[i, j] = top_z 
                else:
                    top_z = self.height_samples[i, j] * self.terrain.vertical_scale
                if visualize:
                    positions.append(Gf.Vec3f(cell_x, cell_y, float(top_z)))
                    proto_indices.append(prototype_index)












                        def _init_command_distribution(self):
        unique_types = torch.unique(self.terrain_types).cpu().tolist()
        self.category_names = [f"terrain_{t}" for t in unique_types]
        self.type2idx      = {t: i for i, t in enumerate(unique_types)}
        self.curricula = [
            RewardThresholdCurriculum(
                seed = self._cfg["commands"]["curriculum_seed"],
                x_vel=(self.limit_vel_x[0],   self.limit_vel_x[1],   self._cfg["commands"]["num_bins_vel_x"]),
                y_vel=(self.limit_vel_y[0],   self.limit_vel_y[1],   self._cfg["commands"]["num_bins_vel_y"]),
                yaw_vel=(self.limit_vel_yaw[0], self.limit_vel_yaw[1], self._cfg["commands"]["num_bins_vel_yaw"]),
            )
            for _ in unique_types
        ]
        
        self.env_command_categories = torch.as_tensor(
        [self.type2idx[int(t)] for t in self.terrain_types.cpu()],
        dtype=torch.long,
        )        
        self.env_command_bins = np.zeros(self.num_envs, dtype=np.int64)

        # 4. Initialise every curriculum’s range once
        low  = np.array([self.command_x_range[0], self.command_y_range[0], self.command_yaw_range[0]])
        high = np.array([self.command_x_range[1], self.command_y_range[1], self.command_yaw_range[1]])
        for cur in self.curricula:
            cur.set_to(low=low, high=high)


    def _resample_commands(self, env_ids):
        """
        Resample (x vel, y vel, yaw vel) commands for the envs in `env_ids`.
        Curriculum progress is now estimated from the *last* entry of the
        tracking-history buffers instead of the aggregated episode‐sums.
        """
        if len(env_ids) == 0:
            return
        # Episode statistics for the selected envs ------------------------
        lin_idx = (self.tracking_lin_vel_x_history_idx[env_ids] - 1) % self.tracking_history_len
        ang_idx = (self.tracking_ang_vel_x_history_idx[env_ids] - 1) % self.tracking_history_len
        len_idx = (self.ep_length_history_idx      [env_ids] - 1) % self.tracking_history_len

        last_lin = self.tracking_lin_vel_x_history[env_ids, lin_idx]             # (B,)
        last_ang = self.tracking_ang_vel_x_history[env_ids, ang_idx]
        last_len = self.ep_length_history      [env_ids, len_idx].clamp_min(1e-6)

        # Curriculum success thresholds
        thr = [
            self._cfg["commands"]["curriculum_thresholds"]["tracking_lin_vel"],
            self._cfg["commands"]["curriculum_thresholds"]["tracking_ang_vel"],
            self._cfg["commands"]["curriculum_thresholds"]["ep_length"]
        ]

        # Split envs by terrain-type curriculum --------------------------
        cats = self.env_command_categories[env_ids.cpu()]                        # (B,)
        for cur_idx in torch.unique(cats):
            mask = (cats == cur_idx)
            sub_envs = env_ids[mask]                                             # tensor on device
            if sub_envs.numel() == 0:
                continue
            cur = self.curricula[int(cur_idx)]
            rewards = torch.stack([
                last_lin[mask],                                                  
                last_ang[mask],                                                  
                last_len[mask],                                                  
            ], dim=1).mean(0).tolist()
            # Let the curriculum object update its bin statistics
            old_bins = self.env_command_bins[sub_envs.cpu().numpy()]
            cur.update(old_bins, rewards, thr,
                            local_range=np.array([0.55, 0.55, 0.55]))

            # Sample new commands & assign them
            new_cmds, new_bins = cur.sample(batch_size=sub_envs.numel())
            self.env_command_bins     [sub_envs.cpu().numpy()] = new_bins
            self.commands[sub_envs, :3] = torch.tensor(new_cmds[:, :3],
                                                            device=self.device)
            











                        # teleport_env_ids = out_of_bounds.nonzero(as_tuple=False).flatten()
            # if teleport_env_ids.numel() > 0:
            #     self.base_pos[teleport_env_ids] = self.base_init_state[0:3]
            #     self.base_pos[teleport_env_ids, 0:3] += self.env_origins[teleport_env_ids]
            #     self.base_pos[teleport_env_ids, 0:2] += torch_rand_float(-1., 1., (len(teleport_env_ids), 2), device=self.device)
            #     indices = teleport_env_ids.to(dtype=torch.int32)
            #     self._anymals.set_world_poses(
            #     positions=self.base_pos[teleport_env_ids].clone(), orientations=self.base_quat[teleport_env_ids].clone(), indices=indices
            # )



















try:
    import isaacsim  # isort: skip
except:
    pass

from omni.isaac.kit import SimulationApp   -  from isaacsim import SimulationApp
from omni.isaac.core.articulations import ArticulationView - from isaacsim.core.prims import Articulation
from omni.isaac.core.prims import RigidPrimView - from isaacsim.core.prims import RigidPrim
import omni 
from omni.isaac.core.articulations import Articulation - from isaacsim.core.prims import SingleArticulation

from omni.isaac.core.robots.robot import Robot - 

from omni.isaac.core.utils.stage import add_reference_to_stage - from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path - from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.torch.maths import set_seed - from isaacsim.core.utils.torch.maths import set_seed
from omni.isaac.core.simulation_context import SimulationContext - from isaacsim.core.api import World, SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path, find_matching_prim_paths, is_prim_path_valid - from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage - from isaacsim.core.utils.stage import get_current_stage

from omni.isaac.core.prims.soft.particle_system import ParticleSystem - from isaacsim.core.prims import ParticleSystem, SingleParticleSystem, SingleXFormPrim
from omni.isaac.core.prims.soft.particle_system_view import ParticleSystemView
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux, Sdf, Gf, UsdShade, Vt
from omni.physx.scripts import physicsUtils, particleUtils

import omni.kit.commands
from omni.isaac.core.scenes.scene import Scene - from isaacsim.core.api.scenes.scene import Scene
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter - from isaacsim.replicator.writers import PytorchWriter

from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener - - from isaacsim.replicator.writers import PytorchListener
import omni.replicator.core as rep - 
from omni.isaac.core.utils.prims import define_prim - from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction - from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.cloner import GridCloner - from isaacsim.core.cloner import GridCloner
import omni.client
from omni.isaac.core.utils.extensions import enable_extension - from isaacsim.core.utils.extensions import enable_extension
 import omni.ui 
from omni.isaac.core.prims import XFormPrim - - from isaacsim.core.prims import ParticleSystem, SingleParticleSystem, SingleXFormPrim