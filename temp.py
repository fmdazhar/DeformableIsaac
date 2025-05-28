else:
            point_instancer = UsdGeom.PointInstancer.Get(self._stage, particle_point_instancer_path)            
            
            existing_positions = point_instancer.GetPositionsAttr().Get()
            existing_velocities = point_instancer.GetVelocitiesAttr().Get()

            # Convert Python lists -> Vt.Vec3fArray (new data)
            new_positions = Vt.Vec3fArray(positions)
            new_velocities = Vt.Vec3fArray(velocities)

            appended_positions = Vt.Vec3fArray(list(existing_positions) + list(new_positions))
            appended_velocities = Vt.Vec3fArray(list(existing_velocities) + list(new_velocities))

            # Re-set the attributes on the same instancer
            point_instancer.GetPositionsAttr().Set(appended_positions)
            point_instancer.GetVelocitiesAttr().Set(appended_velocities)

            # Also update the prototype indices if necessary.
            existing_proto = list(point_instancer.GetProtoIndicesAttr().Get() or [])
            new_proto = [0] * len(new_positions)
            point_instancer.GetProtoIndicesAttr().Set(existing_proto + new_proto)

            # IMPORTANT: Reconfigure the particle set so that the simulation recalculates
            # properties such as mass based on the updated number of particles.
            particleUtils.configure_particle_set(
                point_instancer.GetPrim(),
                particle_system_path,
                self._particle_cfg[system_name]["particle_grid_self_collision"],
                self._particle_cfg[system_name]["particle_grid_fluid"],
                self._particle_cfg[system_name]["particle_grid_particle_group"],
                self._particle_cfg[system_name]["particle_grid_particle_mass"] * len(appended_positions),  # update mass based on total count
                self._particle_cfg[system_name]["particle_grid_density"],
            )
            print(f"[INFO] Appended {len(new_positions)} Particles to {particle_point_instancer_path}")
            key = (level, system_name)
            self.initial_particle_positions[key] = Vt.Vec3fArray(appended_positions)
            self.particle_counts[key] = len(appended_positions)
            self.current_particle_positions[key] = Vt.Vec3fArray(appended_positions)
            # Increment the total_particles counter
            self.total_particles += len(new_positions)    




# Aggregate cached particle positions for this level
all_pos = []
for system_name in system_dict.keys():
    key = (lvl, system_name)
    cached = self.current_particle_positions.get(key)
    if cached:
        all_pos.append(np.array([[p.x, p.y, p.z] for p in cached]))

if not all_pos:
    continue


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



# def update_terrain_level(self, env_ids):
        
    #     if self.init_done:
    #         tracking_lin_vel_high = self.terrain_curriculum_cfg["tracking_lin_vel_high"]* self.reward_scales["tracking_lin_vel"] / self.dt
    #         tracking_ang_vel_high = self.terrain_curriculum_cfg["tracking_ang_vel_high"] * self.reward_scales["tracking_ang_vel"] / self.dt
    #         tracking_episode_length = self.terrain_curriculum_cfg["tracking_eps_length_high"]
    #         full_hist  = self.tracking_lin_vel_x_history_full       # Bool[|env_ids|]
    #         at_cap    = self.terrain_levels == self.unlocked_levels       # Bool[|env_ids|]
    #         not_solved = self.unlocked_levels < self.max_terrain_level + 1
    #         mask      = at_cap[env_ids] & not_solved[env_ids] & full_hist[env_ids]
    #         valid_envs  = env_ids[mask]                                    # maybe empty

    #         if valid_envs.numel():                                             # guard for speed
    #             tracking_lin_vel_x_mean = self.tracking_lin_vel_x_history[valid_envs].mean(dim=1)  # Float[valid_envs]
    #             tracking_ang_vel_x_mean = self.tracking_ang_vel_x_history[valid_envs].mean(dim=1)  # Float[valid_envs]
    #             tracking_episode_length_mean = self.ep_length_history[valid_envs].mean(dim=1)  # Float[valid_envs]
    #             promote_mask = (tracking_lin_vel_x_mean > tracking_lin_vel_high) & \
    #                (tracking_ang_vel_x_mean > tracking_ang_vel_high) & \
    #                (tracking_episode_length_mean > tracking_episode_length)
    #             promote_envs = valid_envs[promote_mask]                         # again maybe empty

    #             if promote_envs.numel():
    #                 self.unlocked_levels[promote_envs] = torch.clamp(
    #                 self.unlocked_levels[promote_envs] + 1,
    #                 max=self.max_terrain_level + 1
    #                 )

    #                 unlocked_cap = torch.clamp(
    #                     self.unlocked_levels[env_ids],
    #                     max=self.max_terrain_level)  

    #                 span = unlocked_cap - self.min_terrain_level  
    #                 u = torch.rand_like(span, dtype=torch.float)
    #                 power = 0
    #                 offset = torch.floor((span + 1).float() * u.pow(1.0 / (power + 1.0))).long()
    #                 old_levels = self.terrain_levels[env_ids]
    #                 new_levels = offset + self.min_terrain_level
    #                 self.terrain_levels[env_ids] = new_levels
    #                 changed_mask = new_levels != old_levels
    #                 if changed_mask.any():
    #                     self._clear_tracking_history(env_ids[changed_mask])

    #     self.terrain_levels[env_ids] = torch.where(self.unlocked_levels[env_ids]>self.max_terrain_level,
    #                                         torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
    #                                         torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        
    #     new_levels = self.terrain_levels[env_ids]
    #     unique_levels, inverse_idx = torch.unique(new_levels, return_inverse=True)
    #     for i, lvl in enumerate(unique_levels):
    #         # Get which envs in env_ids map to this terrain level
    #         group = env_ids[inverse_idx == i]
    #         if group.numel() == 0:
    #             continue

    #         # Rows for this level
    #         candidate_indices = self._terrains_by_level[lvl.item()]
    #         n_envs   = group.shape[0]
    #         n_cands = candidate_indices.shape[0]
    #         idxs = torch.arange(n_envs, device=self.device) % n_cands
    #         chosen_rows = candidate_indices[idxs]
    #         self.bx_start[group] = self.terrain_details[chosen_rows, 10]
    #         self.bx_end[group]   = self.terrain_details[chosen_rows, 11]
    #         self.by_start[group] = self.terrain_details[chosen_rows, 12]
    #         self.by_end[group]   = self.terrain_details[chosen_rows, 13]
    #         self.compliance[group]  = self.terrain_details[chosen_rows, 6].bool()
    #         self.system_idx[group]  = self.terrain_details[chosen_rows, 7].long()
    #         self.terrain_types[group] = self.terrain_details[chosen_rows, 4].long()
    #         rows = self.terrain_details[chosen_rows, 2].long()
    #         cols = self.terrain_details[chosen_rows, 3].long()
    #         self.env_origins[group] = self.terrain_origins[rows, cols]

    #     # Update compliance and stored PBD parameters for these newly changed envs
    #     self.set_compliance(env_ids)
    #     self.store_pbd_params(env_ids) 






        # def update_terrain_level(self, env_ids):
    #     if self.init_done:
    #         current_max = int(self.unlocked_levels.max().item())
    #         at_max = (self.unlocked_levels == current_max)
    #         full_hist = self.tracking_lin_vel_x_history_full & self.tracking_ang_vel_x_history_full & self.ep_length_history_full
    #         if not full_hist[at_max].all():
    #             return

    #         valid_envs = torch.arange(self.num_envs, device=self.device)[at_max]
    #         any_promoted = False
    #         lin_thr = self.terrain_curriculum_cfg["tracking_lin_vel_high"] * self.reward_scales["tracking_lin_vel"] / self.dt
    #         ang_thr = self.terrain_curriculum_cfg["tracking_ang_vel_high"] * self.reward_scales["tracking_ang_vel"] / self.dt
    #         len_thr = self.terrain_curriculum_cfg["tracking_eps_length_high"]

    #         # Evaluate per-level performance
    #         for lvl in [current_max]:
    #             mask = (self.terrain_levels == lvl) 
    #             max_mask = mask[at_max]
    #             lvl_envs = torch.nonzero(mask, as_tuple=False).flatten()
    #             if lvl_envs.numel() == 0:
    #                 continue
    #             r_lin = self.tracking_lin_vel_x_history[lvl_envs].mean()
    #             r_ang = self.tracking_ang_vel_x_history[lvl_envs].mean()
    #             r_len = self.ep_length_history[lvl_envs].mean()
    #             if (r_lin > lin_thr) and (r_ang > ang_thr) and (r_len > len_thr):
    #                 # Promote all these envs to next level
    #                 next_lvl = min(lvl + 1, self.max_terrain_level)
    #                 self.unlocked_levels[lvl_envs] = next_lvl
    #                 self.target_levels[lvl_envs] = next_lvl
    #                 any_promoted = True

    #         if not any_promoted:
    #             return

    #         # Resample target levels randomly within unlocked span
    #         span = self.unlocked_levels - self.min_terrain_level
    #         u = torch.rand_like(span, dtype=torch.float)
    #         offset = torch.floor((span + 1).float() * u).long()
    #         self.target_levels = offset + self.min_terrain_level

    #     # Common assignment for pending envs
    #     pending = self.terrain_levels[env_ids] != self.target_levels[env_ids]
    #     pending_envs = env_ids[pending]
    #     if pending_envs.numel() > 0:
    #         old = self.terrain_levels[pending_envs].clone()
    #         new = self.target_levels[pending_envs]
    #         self.terrain_levels[pending_envs] = new
    #         # Clear history for those that actually changed
    #         changed = new != old
    #         if changed.any():
    #             self._clear_tracking_history(pending_envs[changed])

    #     # Assign terrain details for each group
    #     unique_lvls, inv = torch.unique(self.terrain_levels[env_ids], return_inverse=True)
    #     for i, lvl in enumerate(unique_lvls):
    #         group = env_ids[inv == i]
    #         if group.numel() == 0:
    #             continue
    #         rows = self._terrains_by_level[int(lvl.item())]
    #         n_cands = rows.shape[0]
    #         idxs = torch.arange(group.shape[0], device=self.device) % n_cands
    #         chosen = rows[idxs]
    #         # Update block boundaries and properties
    #         self.bx_start[group] = self.terrain_details[chosen, 10]
    #         self.bx_end[group]   = self.terrain_details[chosen, 11]
    #         self.by_start[group] = self.terrain_details[chosen, 12]
    #         self.by_end[group]   = self.terrain_details[chosen, 13]
    #         self.compliance[group] = self.terrain_details[chosen, 6].bool()
    #         self.system_idx[group] = self.terrain_details[chosen, 7].long()
    #         self.terrain_types[group] = self.terrain_details[chosen, 4].long()
    #         rc = self.terrain_origins[ self.terrain_details[chosen,2].long(), self.terrain_details[chosen,3].long() ]
    #         self.env_origins[group] = rc

    #     # Update compliance and PBD params
    #     self.set_compliance(env_ids)
    #     self.store_pbd_params(env_ids)



        def _init_command_ranges_by_terrain(self):
        """
        Build a lookup table that holds per-terrain-TYPE command ranges.

        Shape: [num_terrain_types, 6]  
            (min_x, max_x,  min_y, max_y,  min_yaw, max_yaw)

        The table is stored in  `self.command_ranges_by_terrain`
        and initialised with the *global* command ranges that come from
        the YAML (self.command_*_range).
        """
        unique_ids = torch.unique(self.terrain_details[:, 4]).long()
        self._id2row = {int(t.item()): i for i, t in enumerate(unique_ids)}
        
        base = torch.tensor(
            [ self.command_x_range[0], self.command_x_range[1],
            self.command_y_range[0], self.command_y_range[1],
            self.command_yaw_range[0], self.command_yaw_range[1] ],
            dtype=torch.float, device=self.device
        )

        # replicate →  [T, 6]
        self.command_ranges_by_terrain = base.unsqueeze(0).repeat(len(unique_ids), 1).clone()

    
    # def _resample_commands(self, env_ids):
    #     """
    #     Resample (x vel, y vel, yaw vel) commands for the envs in `env_ids`.
    #     Curriculum progress is now estimated from the *last* entry of the
    #     tracking-history buffers instead of the aggregated episode‐sums.
    #     """
    #     if len(env_ids) == 0:
    #         return
            
    #     self.env_command_categories = torch.as_tensor(
    #         [self.type2idx[int(t)] for t in self.terrain_types.cpu()],
    #         dtype=torch.long,
    #         device=self.device,
    #     )
    #     window_size = self.command_curriculum_cfg["tracking_length"]
    #     last = torch.stack(
    #         [ (self.tracking_lin_vel_x_history_idx - k - 1) % self.tracking_history_len
    #         for k in range(window_size) ],
    #         dim=0
    #     ) 

    #     # Curriculum success thresholds
    #     thr = [
    #         self.command_curriculum_cfg["tracking_lin_vel_high"] * self.reward_scales["tracking_lin_vel"] / self.dt,
    #         self.command_curriculum_cfg["tracking_ang_vel_high"] * self.reward_scales["tracking_ang_vel"] / self.dt,
    #         self.command_curriculum_cfg["tracking_eps_length_high"]
    #     ]

    #     # Split envs by terrain-type curriculum --------------------------
    #     cats = self.env_command_categories[env_ids.cpu()]                        
    #     for cur_idx in torch.unique(cats):
    #         mask = (cats == cur_idx)
    #         sub_envs = env_ids[mask]                                            
    #         if sub_envs.numel() == 0:
    #             continue
    #         # ---- pull the last-window rewards & episode lengths, shape: [W, B]
    #         idx = last[:, sub_envs]
    #         tracking_lin_vel_x_history  = self.tracking_lin_vel_x_history[sub_envs.unsqueeze(0), idx]
    #         tracking_ang_vel_x_history = self.tracking_ang_vel_x_history[sub_envs.unsqueeze(0), idx]
    #         leng  = self.ep_length_history        [sub_envs.unsqueeze(0), idx]

    #         # ignore if any of the last-window slots are still zero
    #         if not ((tracking_lin_vel_x_history != 0).all() and (tracking_ang_vel_x_history != 0).all() and (leng != 0).all()):
    #             continue

    #         r_mean_lin = tracking_lin_vel_x_history.mean().item()
    #         r_mean_ang = tracking_ang_vel_x_history.mean().item()
    #         l_mean = leng.float().mean().item()

    #         rewards = [r_mean_lin, r_mean_ang, l_mean]

    #         # Let the curriculum object update its bin statistics
    #         cur = self.curricula[int(cur_idx)]
    #         old_bins = self.env_command_bins[sub_envs.cpu().numpy()]
    #         cur.update(old_bins, rewards, thr,
    #                         local_range=np.array([0.55, 0.55, 0.55]))

    #         # Sample new commands & assign them
    #         new_cmds, new_bins = cur.sample(batch_size=sub_envs.numel())
    #         changed_mask = new_bins != old_bins
    #         if changed_mask.any():                               # at least one env moved
    #             changed_envs = sub_envs[torch.from_numpy(changed_mask)
    #                                     .to(sub_envs.device)]
    #             self._clear_tracking_history(changed_envs)

    #         self.env_command_bins     [sub_envs.cpu().numpy()] = new_bins
    #         cmd_tensor = torch.tensor(
    #             new_cmds[:, :3], dtype=self.commands.dtype, device=self.device
    #         )
    #         self.commands[sub_envs, :3] = cmd_tensor



        # def query_top_particle_positions(self, visualize=False):
    #     if not self.current_particle_positions:
    #         return
    #     cell_scale     = self.terrain.horizontal_scale
    #     border_size    = self.terrain.border_size
    #     half_cell      = cell_scale / 2.0
    #     v_scale        = self.terrain.vertical_scale
    #     vis_positions, vis_proto_idx = [], []
    #     # Iterate each level and its systems
        
    #     for lvl, particles in self.current_particle_positions.items():
    #         if particles.size == 0:
    #             continue
            
    #         env_ids = (self.terrain_levels == lvl).nonzero(as_tuple=False).flatten()
    #         if env_ids.numel() == 0:
    #             continue  # no robots on this level this frame

    #         foot_positions = self.foot_pos.view(self.num_envs, 4, 3)  
    #         if env_ids is not None:
    #             foot_positions  = foot_positions[env_ids]    
    #         N = foot_positions.shape[0]
    #         points = (foot_positions.unsqueeze(2) + self.particle_height_points.view(N, 4, -1, 3)).reshape(N, self._num_particle_height_points, 3)
    #         points += self.terrain.border_size
    #         points = (points / self.terrain.horizontal_scale).long()
    #         px = points[:, :, 0].view(-1)
    #         py = points[:, :, 1].view(-1)
    #         px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
    #         py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

    #         grid_indices = torch.stack((px, py), dim=1)  
    #         uniq_cells   = torch.unique(grid_indices, dim=0)  
    #         cell_x = uniq_cells[:, 0].float() * cell_scale - border_size          # (M,)
    #         cell_y = uniq_cells[:, 1].float() * cell_scale - border_size 

    #         cx_min, cx_max = cell_x - half_cell, cell_x + half_cell               # (M,)
    #         cy_min, cy_max = cell_y - half_cell, cell_y + half_cell

    #         if particles.ndim == 1:
    #             particles = particles.unsqueeze(0)        # safety: (P,3)
    #         pxs, pys = particles[:, 0].unsqueeze(1), particles[:, 1].unsqueeze(1)  # (P,1)

    #         # broadcast comparison → (P,M) boolean mask
    #         mask = (
    #             (pxs >= cx_min) & (pxs < cx_max) &
    #             (pys >= cy_min) & (pys < cy_max)
    #         )

    #         for c in range(uniq_cells.shape[0]):
    #             part_mask = mask[:, c]
    #             if not part_mask.any():
    #                 continue

    #             top_z = torch.minimum(
    #                 particles[part_mask, 2].max(),
    #                 torch.tensor(0.0, device=self.device)
    #             )
    #             i_idx = int(uniq_cells[c, 0])
    #             j_idx = int(uniq_cells[c, 1])

    #             # write back to height-field (round + cast to int)
    #             self.height_samples[i_idx, j_idx] = int(torch.round(top_z / v_scale).item())

    #             # optional visualisation bookkeeping
    #             if visualize:
    #                 vis_positions.append(Gf.Vec3f(cell_x[c].item(),
    #                                             cell_y[c].item(),
    #                                             float(self.height_samples[i_idx, j_idx])))
    #                 vis_proto_idx.append(0)

    #     if visualize and vis_positions:
    #         if not hasattr(self, "particle_height_point_instancer"):
    #             self._init_particle_height_instancer()

    #         self.particle_height_point_instancer.CreatePositionsAttr()   \
    #             .Set(Vt.Vec3fArray(vis_positions))
    #         self.particle_height_point_instancer.CreateProtoIndicesAttr()\
    #             .Set(Vt.IntArray(vis_proto_idx))



    def update_command_curriculum(self, env_ids: torch.Tensor) -> None:
        """
        Expand / contract the allowable **x-linear-velocity** range *separately*
        for every terrain **type** that is present in `env_ids`.

        Strategy – identical to your previous implementation:
        • look at the last `window_size` episodes for every env  
        • if mean tracking-reward & episode-length are high ⇒ widen range  
        • if they are low                                    ⇒ narrow range
        """
        if not env_ids.numel():
            return

        delta_lin   = self.command_curriculum_cfg["delta_lin_vel_x"]
        window_size = self.command_curriculum_cfg["tracking_length"]
        tracking_lin_vel_high   =  self.command_curriculum_cfg["tracking_lin_vel_high"] * self.reward_scales["tracking_lin_vel"] / self.dt
        tracking_lin_vel_low   = self.command_curriculum_cfg["tracking_lin_vel_low"] * self.reward_scales["tracking_lin_vel"] / self.dt
        tracking_ang_vel_high   =  self.command_curriculum_cfg["tracking_ang_vel_high"] * self.reward_scales["tracking_ang_vel"] / self.dt
        tracking_ang_vel_low   = self.command_curriculum_cfg["tracking_ang_vel_low"] * self.reward_scales["tracking_ang_vel"] / self.dt
        len_high      = self.command_curriculum_cfg["tracking_eps_length_high"]
        len_low       = self.command_curriculum_cfg["tracking_eps_length_low"]

        # ── gather helper indices for “last N” circular-buffer access
        last = torch.stack(
            [ (self.tracking_lin_vel_x_history_idx - k - 1) % self.tracking_history_len
            for k in range(window_size) ],
            dim=0
        )                                                      # [N, num_envs]

        # ── process every terrain TYPE seen in env_ids
        terr_types    = self.terrain_types[env_ids]            # (len(env_ids),)
        unique_types  = torch.unique(terr_types)

        for t in unique_types:
            mask = terr_types == t
            these_envs = env_ids[mask]                         # 1-D tensor
            if not these_envs.numel():
                continue

            # ---- pull the last-window rewards & episode lengths, shape: [W, B]
            idx = last[:, these_envs]
            tracking_lin_vel_x_history  = self.tracking_lin_vel_x_history[these_envs.unsqueeze(0), idx]
            tracking_ang_vel_x_history = self.tracking_ang_vel_x_history[these_envs.unsqueeze(0), idx]
            leng  = self.ep_length_history        [these_envs.unsqueeze(0), idx]

            # ignore if any of the last-window slots are still zero
            if not ((tracking_lin_vel_x_history != 0).all() and (tracking_ang_vel_x_history != 0).all() and (leng != 0).all()):
                continue

            r_mean_lin = tracking_lin_vel_x_history.mean()
            r_mean_ang = tracking_ang_vel_x_history.mean()
            l_mean = leng.float().mean()

            # ---- choose direction
            if (r_mean_lin >= tracking_lin_vel_high) and (r_mean_ang >= tracking_ang_vel_high) and (l_mean >= len_high):
                direction = +1           # become harder ⇒ widen range
            # elif (r_mean_lin < tracking_lin_vel_low) or (r_mean_ang < tracking_ang_vel_low) or (l_mean <  len_low):
            #     direction = -1           # become easier ⇒ shrink range
            else:
                direction = 0

            if direction == 0:
                continue

            # ---- apply the adjustment **clamped** to absolute limits
            row = self._id2row[int(t.item())]
            cur_min = self.command_ranges_by_terrain[row, 0]
            cur_max = self.command_ranges_by_terrain[row, 1]

            if direction == +1:
                new_min = torch.clamp(cur_min - delta_lin, min=self.limit_vel_x[0])
                new_max = torch.clamp(cur_max + delta_lin, max=self.limit_vel_x[1])
            else:
                new_min = torch.clamp(cur_min + delta_lin, max=self.command_x_range[0])
                new_max = torch.clamp(cur_max - delta_lin, min=self.command_x_range[1])

            # ––– only act if something really changed ––––––––––––––––––––––––––––––––
            changed = (not torch.isclose(new_min, cur_min)) or (not torch.isclose(new_max, cur_max))
            if changed:
                self.command_ranges_by_terrain[row, 0] = new_min
                self.command_ranges_by_terrain[row, 1] = new_max
                self._clear_tracking_history(these_envs)


    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """
        Resample the x-linear-velocity command for every env in `env_ids` that
        has a terrain level > 0. The new command is sampled from the range
        defined by the terrain type of the env.
        """
        if not env_ids.numel():
            return

        # ---- get the terrain TYPE for each env in `env_ids`
        terr_types = self.terrain_types[env_ids]                # (len(env_ids),)
        unique_types = torch.unique(terr_types)                 # (len(unique_types),)

        # ---- resample the x-linear-velocity command for each TYPE
        for t in unique_types:
            mask = terr_types == t
            these_envs = env_ids[mask]                         # 1-D tensor
            if not these_envs.numel():
                continue

            row = self._id2row[int(t.item())]
            min_x, max_x = self.command_ranges_by_terrain[row, 0:2]

            self.commands[these_envs, 0] = torch.rand(
                len(these_envs), device=self.device, dtype=torch.float) * (max_x - min_x) + min_x    
            
            
            def update_terrain_level(self, env_ids):

        if not self.init_done or not self.curriculum or self.test:
            # do not change on initial reset
            return

        tracking_lin_vel_high = self.terrain_curriculum_cfg["tracking_lin_vel_high"]* self.reward_scales["tracking_lin_vel"] / self.dt
        tracking_ang_vel_high = self.terrain_curriculum_cfg["tracking_ang_vel_high"] * self.reward_scales["tracking_ang_vel"] / self.dt
        tracking_episode_length = self.terrain_curriculum_cfg["tracking_eps_length_high"]
        
        full_hist  = self.tracking_lin_vel_x_history_full 
        current_max = self.unlocked_levels.max()     
        at_cap    = self.terrain_levels == current_max   
        
        envs = torch.arange(self.num_envs, device=self.device)
        mask      = at_cap  & full_hist 
        valid_envs  = envs[mask] 
        
        tracking_lin_vel_x_mean = self.tracking_lin_vel_x_history[valid_envs].mean()  
        tracking_ang_vel_x_mean = self.tracking_ang_vel_x_history[valid_envs].mean()  
        tracking_episode_length_mean = self.ep_length_history[valid_envs].mean()  

        cond = (tracking_lin_vel_x_mean   > tracking_lin_vel_high)  & \
        (tracking_ang_vel_x_mean   > tracking_ang_vel_high)  & \
        (tracking_episode_length_mean > tracking_episode_length)

        if cond:
            self.unlocked_levels = torch.clamp(
            self.unlocked_levels + 1,
            max=self.max_terrain_level + 1
            )               
            next_lvl = torch.clamp(current_max + 1, max=self.max_terrain_level)
            self.target_levels.fill_(next_lvl)

            # Resample the target terrain level
            span = self.target_levels - self.min_terrain_level  
            u = torch.rand_like(span, dtype=torch.float)
            power = 0
            offset = torch.floor((span + 1).float() * u.pow(1.0 / (power + 1.0))).long()
            new_targets = offset + self.min_terrain_level
            self.target_levels = new_targets

        pending_mask = self.terrain_levels[env_ids] != self.target_levels[env_ids]
        pending_envs = env_ids[pending_mask]
        new_levels = self.target_levels[pending_envs]
        old_levels = self.terrain_levels[pending_envs]
        self.terrain_levels[pending_envs] = new_levels
        changed_mask = new_levels != old_levels
        if changed_mask.any():
            self._clear_tracking_history(pending_envs[changed_mask])






    def update_terrain_level(self, env_ids):

        if not self.init_done or not self.curriculum or self.test:
            # do not change on initial reset
            return

        tracking_lin_vel_high = self.terrain_curriculum_cfg["tracking_lin_vel_high"]* self.reward_scales["tracking_lin_vel"] / self.dt
        tracking_ang_vel_high = self.terrain_curriculum_cfg["tracking_ang_vel_high"] * self.reward_scales["tracking_ang_vel"] / self.dt
        tracking_episode_length = self.terrain_curriculum_cfg["tracking_eps_length_high"]
        
        full_hist  = self.tracking_lin_vel_x_history_full 
        current_max = self.unlocked_levels.max()     
        at_cap    = self.terrain_levels == current_max   
        
        envs = torch.arange(self.num_envs, device=self.device)
        mask      = at_cap  & full_hist 
        valid_envs  = envs[mask] 
        
        tracking_lin_vel_x_mean = self.tracking_lin_vel_x_history[valid_envs].mean()  
        tracking_ang_vel_x_mean = self.tracking_ang_vel_x_history[valid_envs].mean()  
        tracking_episode_length_mean = self.ep_length_history[valid_envs].mean()  

        cond = (tracking_lin_vel_x_mean   > tracking_lin_vel_high)  & \
        (tracking_ang_vel_x_mean   > tracking_ang_vel_high)  & \
        (tracking_episode_length_mean > tracking_episode_length)

        if cond:
            self.unlocked_levels = torch.clamp(
            self.unlocked_levels + 1,
            max=self.max_terrain_level + 1
            )               
            next_lvl = torch.clamp(current_max + 1, max=self.max_terrain_level)
            self.target_levels.fill_(next_lvl)

            # Resample the target terrain level
            span = self.target_levels - self.min_terrain_level  
            u = torch.rand_like(span, dtype=torch.float)
            power = 0
            offset = torch.floor((span + 1).float() * u.pow(1.0 / (power + 1.0))).long()
            new_targets = offset + self.min_terrain_level
            self.target_levels = new_targets

        pending_mask = self.terrain_levels[env_ids] != self.target_levels[env_ids]
        pending_envs = env_ids[pending_mask]
        new_levels = self.target_levels[pending_envs]
        old_levels = self.terrain_levels[pending_envs]
        self.terrain_levels[pending_envs] = new_levels
        changed_mask = new_levels != old_levels
        if changed_mask.any():
            self._clear_tracking_history(pending_envs[changed_mask])




                                
                    # # --- auto‑reset on excessive particle loss -----------------
                    # key = (lvl, system_name, grid_key)
                    # init_cnt = self.particle_counts.get(key)
                    # lx0, lx1, ly0, ly1 = self.bounds.get(int(grid_key))
                    # # lz0 = -2
                    # # lz1 = 2
                    # inside = (pts[:,0] >= lx0) & (pts[:,0] < lx1) & (pts[:,1] >= ly0) & (pts[:,1] < ly1)  #& (pts[:, 2] >= lz0) & (pts[:, 2] < lz1)
                    # filtered = pts[inside]
                    # valid_count = filtered.shape[0]
                    # needs_reset = self._particle_cfg["reset_check"] and init_cnt is not None and valid_count < self._particle_cfg["particle_reset_threshold"] * init_cnt
                    # if needs_reset:
                    #     carb.log_warn(
                    #         f"[PBD]Resetting {system_name}@level{lvl}: "
                    #         f"{len(positions)}/{init_cnt} particles remain (<80%)."
                    #     )
                    #     self._reset_particle_grid(lvl, system_name, grid_key)
                    #     init_pos = self.initial_particle_positions[(lvl, system_name, grid_key)]
                    #     pts      = torch.tensor(init_pos, dtype=torch.float32, device=self.device)