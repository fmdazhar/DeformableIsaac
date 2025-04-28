import math
import numpy as np
import torch
import omni
import carb

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path, find_matching_prim_paths, is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.a1 import A1
from omniisaacgymenvs.robots.articulations.views.a1_view import A1View
from omni.isaac.core.prims import RigidPrimView
from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from omniisaacgymenvs.tasks.utils.curriculum import RewardThresholdCurriculum

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux, Sdf, Gf, UsdShade, Vt

from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.prims.soft.particle_system_view import ParticleSystemView
from omni.physx.scripts import physicsUtils, particleUtils
import omni.kit.commands


class AnymalTerrainTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.height_samples = None
        self.terrain_details = None
        self.custom_origins = False
        self.init_done = False
        self._env_spacing = 0.0

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._device = self._cfg["sim_device"]
        self.update_config(sim_config)

        self._num_actions = 12
        self._num_proprio = 52 #188 #3 + 3 + 3 + 3 + 12 + 12 + 12 + 4 
        self._num_privileged_observations = None
        # self._num_priv = 28 #4 + 4 + 4 + 1 + 3 + 12 
        self._obs_history_length = 10  # e.g., 3, 5, etc.
        # If measure_heights is True, we add that to the final observation dimension
        if self.measure_heights:
            self._num_height_points = 36 #140
        else:
            self._num_height_points = 0
        self._num_obs_history = self._obs_history_length * (self._num_proprio + self._num_height_points)

        # Then the final observation dimension is:
        self._num_observations = self._num_obs_history \
                                + self._num_priv \
                                + self._num_proprio \
                                + self._num_height_points

        RLTask.__init__(self, name, env)

        self._num_train_envs = self.num_envs

        if self.measure_heights:
            self.height_points = self.init_height_points()
            self.particle_height_points = self.init_particle_height_points()
            self.measured_heights = torch.zeros(
                (self.num_envs, self._num_height_points),
                dtype=torch.float,
                device=self.device
            )
        else:
            self.measured_heights = None
        self.debug_heights = True

        # Initialize dictionaries to track created particle systems and materials
        self.created_particle_systems = {}
        self.created_materials = {}
        self.particle_instancers_by_level = {}
        self._terrains_by_level = {}  # dictionary: level -> (tensor of row indices)
        self.total_particles = 0    # Initialize a counter for total particles

        # joint positions offsets
        self.default_dof_pos = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False
        )

        return

    def update_config(self, sim_config):


        # normalization
        self.clip_obs = self._task_cfg["env"].get("clipObservations", np.Inf)
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["learn"]["contactForceScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward params
        self.base_height_target = self._task_cfg["env"]["learn"]["baseHeightTarget"]
        self.soft_dof_pos_limit = self._task_cfg["env"]["learn"]["softDofPositionLimit"]
        self.soft_dof_vel_limit = self._task_cfg["env"]["learn"]["softDofVelLimit"]
        self.soft_torque_limit = self._task_cfg["env"]["learn"]["softTorqueLimit"]
        self.tracking_sigma = self._task_cfg["env"]["learn"]["trackingSigma"]
        self.max_contact_force = self._task_cfg["env"]["learn"]["maxContactForce"]
        self.only_positive_rewards = self._task_cfg["env"]["learn"]["onlyPositiveRewards"]
        
        # reward scales
        self.reward_scales = self._task_cfg["env"]["learn"]["scales"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["commands"]["lin_vel_x"]
        self.command_y_range = self._task_cfg["env"]["commands"]["lin_vel_y"]
        self.command_yaw_range = self._task_cfg["env"]["commands"]["ang_vel_yaw"]
        self.limit_vel_x = self._task_cfg["env"]["commands"]["limit_vel_x"]
        self.limit_vel_y = self._task_cfg["env"]["commands"]["limit_vel_y"]
        self.limit_vel_yaw = self._task_cfg["env"]["commands"]["limit_vel_yaw"]
        self.vel_curriculum = self._task_cfg["env"]["commands"]["VelocityCurriculum"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        #randomization
        self.friction_range = self._task_cfg["env"]["randomizationRanges"]["frictionRange"]
        self.restitution_range = self._task_cfg["env"]["randomizationRanges"]["restitutionRange"]
        self.added_mass_range = self._task_cfg["env"]["randomizationRanges"]["addedMassRange"]
        self.com_displacement_range = self._task_cfg["env"]["randomizationRanges"]["comDisplacementRange"]
        self.motor_strength_range = self._task_cfg["env"]["randomizationRanges"]["motorStrengthRange"]
        self.motor_offset_range = self._task_cfg["env"]["randomizationRanges"]["motorOffsetRange"]
        self.Kp_factor_range = self._task_cfg["env"]["randomizationRanges"]["KpFactorRange"]
        self.Kd_factor_range = self._task_cfg["env"]["randomizationRanges"]["KdFactorRange"]
        self.gravity_range = self._task_cfg["env"]["randomizationRanges"]["gravityRange"]
        self.stiffness_range = self._task_cfg["env"]["randomizationRanges"]["material_randomization"]["compliance"]["stiffness"]
        self.damping_multiplier_range = self._task_cfg["env"]["randomizationRanges"]["material_randomization"]["compliance"]["damping_multiplier"]
        
        # Control which privileged observations are included
        self.priv_base       = self._task_cfg['env'].get('priv_base', True)
        self.priv_compliance = self._task_cfg['env'].get('priv_compliance', True)
        self.priv_pbd_particle = self._task_cfg['env'].get('priv_pbd_particle', True)
        
        # Calculate the damping range:
        damping_min = self.stiffness_range[0] * self.damping_multiplier_range[0]
        damping_max = self.stiffness_range[1] * self.damping_multiplier_range[1]
        self.damping_range = (damping_min, damping_max)
        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_enabled = self._task_cfg["env"]["learn"]["pushEnabled"]
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.randomize_pbd = self._task_cfg["env"]["randomizationRanges"]["material_randomization"]["particles"]["enabled"]
        self.randomize_pbd_interval = int(self._task_cfg["env"]["randomizationRanges"]["material_randomization"]["particles"]["interval"] / self.dt + 0.5)
        self.randomize_gravity = self._task_cfg["env"]["randomizationRanges"]["randomizeGravity"]
        self.gravity_randomize_interval = int(self._task_cfg["env"]["randomizationRanges"]["gravityRandIntervalSecs"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.teleport_active = self._task_cfg["env"]["terrain"]["teleportActive"]
        self.teleport_buffer = self._task_cfg["env"]["terrain"]["teleportBuffer"]
        self.measure_heights = self._task_cfg["env"]["terrain"]["measureHeights"]

        self.base_threshold = 0.2
        self.thigh_threshold = 0.1

        self._num_envs = self._task_cfg["env"]["numEnvs"]

        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"][
            "staticFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"][
            "dynamicFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"][
            "restitution"
        ]

        self._task_cfg["sim"]["add_ground_plane"] = False
        self._particle_cfg = self._task_cfg["env"]["terrain"]["particles"]

        self.terrain = Terrain(self._task_cfg["env"]["terrain"])
        self.terrain_details = torch.tensor(self.terrain.terrain_details, dtype=torch.float, device=self.device)

        self._particles_active = (self.terrain_details[:, 5] > 0).any().item()
        self._compliance_active = (self.terrain_details[:, 6] > 0).any().item()
        
        if self._particles_active and self.priv_pbd_particle:
            self.init_pbd()
        else:
            self.pbd_range = None
            self.pbd_param_names = []
            self.pbd_parameters = None
            self.pbd_scale = None
            self.pbd_shift = None
            self._pbd_idx = {}
            self.pbd_by_sys = {}

        self._num_priv = (
            (28 if self.priv_base else 0)
        + (8  if (self._compliance_active and self.priv_compliance) else 0)
        + (len(self.pbd_parameters) if self.pbd_parameters is not None else 0)
        )
        print(f"Num priv: {self._num_priv}")
    
    def init_pbd(self):
        """
        Collect the [min, max] ranges of every PBD material parameter that is
        randomised for the *active* particle systems in the task YAML and store
        them in `self.pbd_range` (shape: N×2, dtype=float32, on `self.device`).

        After this you can call:
            self.pbd_scale, self.pbd_shift = self.get_scale_shift(self.pbd_range)
        """
        active_system_ids = {int(s.item())
                         for s in torch.unique(self.terrain_details[:, 7])
                         if s > 0}

        systems_cfg = (
                self._task_cfg["env"]["randomizationRanges"]
                            ["material_randomization"]["particles"]["systems"]
            )

        # merge ranges by parameter name
        merged = {}
        self.pbd_by_sys = {}
        for sys_name in active_system_ids:
            cfg = systems_cfg.get(sys_name, {})
            if not cfg.get("enabled", False):
                continue
            local = []
            for param, rng in cfg.items():
                if not (param.endswith("_range") and isinstance(rng, (list, tuple)) and len(rng) == 2):
                    continue
                base_param = param[:-6]  # e.g. "pbd_material_friction_range" -> "pbd_material_friction"
                local.append((base_param, rng))
                lo, hi = rng
                if base_param in merged:
                    merged_lo, merged_hi = merged[base_param]
                    merged[base_param] = [min(lo, merged_lo), max(hi, merged_hi)]
                else:
                    merged[base_param] = [lo, hi]
            self.pbd_by_sys[sys_name] = local
            
        self.pbd_param_names = sorted(merged.keys())
        self.pbd_range = [merged[k] for k in self.pbd_param_names]
        self.pbd_param_names += ["particles_present", "fluid_present"]
        self.pbd_range += [[0.0, 1.0], [0.0, 1.0]] 

        self.pbd_parameters = torch.zeros((self.num_envs, len(self.pbd_param_names)),
                                           dtype=torch.float32,
                                           device=self.device)
        self._pbd_idx = {name: i for i, name in enumerate(self.pbd_param_names)}


        scale_shift_pairs = [self.get_scale_shift(rng) for rng in self.pbd_range]
        scales, shifts = zip(*scale_shift_pairs)
        self.pbd_scale = torch.tensor(scales, dtype=torch.float32, device=self.device)   # shape: [num_params]
        self.pbd_shift = torch.tensor(shifts, dtype=torch.float32, device=self.device)   # shape: [num_params]

        self._system_is_fluid = {
        sid: float(self._particle_cfg[f"system{sid}"].get("particle_grid_fluid", False))
        for sid in active_system_ids
        }
        
    def _get_noise_scale_vec(self):

        noise_vec = torch.zeros(self._num_proprio, device=self.device, dtype=torch.float)
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:24] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:40] = self._task_cfg["env"]["learn"]["contactForceNoise"] * noise_level * self.contact_force_scale
        noise_vec[40:52] = 0.0  # previous actions

        return noise_vec
        
    def init_height_points(self):
        y = 0.1 * torch.tensor(
            [-1, 0, 1], device=self.device, requires_grad=False
        )
        x = 0.1 * torch.tensor(
            [-1, 0, 1], device=self.device, requires_grad=False
        )
        
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        num_points = grid_x.numel()
        base = torch.stack([grid_x.flatten(), grid_y.flatten(),
                        torch.zeros(num_points, device=self.device)], dim=1)  # (9,3)
        foot_offsets = base.repeat(4, 1)
        self._num_height_points = foot_offsets.shape[0] 
        points = foot_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)        
        
        return points 
                                           
    def init_particle_height_points(self):
        y = 0.1 * torch.tensor(
            [-2, -1, 0, 1, 2], device=self.device, requires_grad=False
        )
        x = 0.1 * torch.tensor(
            [-2, -1, 0, 1, 2], device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        num_points = grid_x.numel()
        base = torch.stack([grid_x.flatten(), grid_y.flatten(),
                        torch.zeros(num_points, device=self.device)], dim=1) 
        foot_offsets = base.repeat(4, 1)
        self._num_particle_height_points = foot_offsets.shape[0] 
        points = foot_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)        
        
        return points

    def _create_trimesh(self, create_mesh=True):
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).float().to(self.device)
        self.terrain_types = [int(l.item()) for l in torch.unique(self.terrain_details[:, 4])]
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
        if create_mesh:
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _init_command_distribution(self, env_ids):
            # new style curriculum
            self.category_names = [f"terrain_{t}" for t in self.terrain_types]
            self.curricula = []
            for category in self.category_names:
                self.curricula += [RewardThresholdCurriculum(seed=self.cfg.commands.curriculum_seed,
                                                x_vel=(self.cfg.commands.limit_vel_x[0],
                                                        self.cfg.commands.limit_vel_x[1],
                                                        self.cfg.commands.num_bins_vel_x),
                                                y_vel=(self.cfg.commands.limit_vel_y[0],
                                                        self.cfg.commands.limit_vel_y[1],
                                                        self.cfg.commands.num_bins_vel_y),
                                                yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                            self.cfg.commands.limit_vel_yaw[1],
                                                            self.cfg.commands.num_bins_vel_yaw),
                                                )]

            self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
            self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
            low = np.array(
                [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
                self.cfg.commands.ang_vel_yaw[0],  ])
            high = np.array(
                [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
                self.cfg.commands.ang_vel_yaw[1],])
            for curriculum in self.curricula:
                curriculum.set_to(low=low, high=high)

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return
        # update curricula based on terminated environment bins and categories
        for i, (_, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]
            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", ]:
                if key in self.episode_sums.keys():
                    task_rewards.append(self.episode_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55,]))

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue
            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)
            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i
            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :3]).to(
            self.device)

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def update_terrain_level(self, env_ids):

        if self.init_done:

            # For each currently represented level, check performance
            current_levels = self.terrain_levels[env_ids]
            unique_levels, inverse_idx = torch.unique(current_levels, return_inverse=True)
            # Threshold for leveling up
            threshold = 0.8

            # Promote envs whose recent tracking velocity exceeds threshold
            for lvl in unique_levels:
                if lvl >= self.highest_level:
                # already at max, optionally randomize or skip
                    continue
                # Indices of env_ids in this level
                mask = (current_levels == lvl)
                level_envs = env_ids[mask]
                if level_envs.numel() == 0:
                    continue
                self.terrain_levels[level_envs] = lvl + 1
                print(f"Leveling up envs {level_envs} from {lvl} to {lvl + 1}")
                # # Compute mean over the stored history
                # mean_reward = self.tracking_lin_vel_x_history[level_envs].mean()

                # # If above threshold and not at max, bump their level
                # if mean_reward > threshold:
                #     self.terrain_levels[level_envs] = lvl + 1

        new_levels = self.terrain_levels[env_ids]
        # over_max  = new_levels > self.highest_level
        # if over_max.any():
        #     envs_to_rand = env_ids[over_max]
        #     self.terrain_levels[envs_to_rand] = torch.randint(
        #         low=0,
        #         high=self.highest_level + 1,
        #         size=(envs_to_rand.shape[0],),
        #         device=self.device,
        #     )

        # # For any envs that have reached highest_level, allow random levels
        # at_max = (self.terrain_levels[env_ids] == self.highest_level)
        # if at_max.any():
        #     max_envs = env_ids[at_max]
        #     # sample random levels between 0 and highest_level inclusive
        #     self.terrain_levels[max_envs] = torch.randint(
        #         low=0,
        #         high=self.highest_level + 1,
        #         size=(len(max_envs),),
        #         device=self.device,
        #     )
        
        unique_levels, inverse_idx = torch.unique(new_levels, return_inverse=True)
        for i, lvl in enumerate(unique_levels):
            # Get which envs in env_ids map to this terrain level
            group = env_ids[inverse_idx == i]
            if group.numel() == 0:
                continue

            # Rows for this level
            candidate_indices = self._terrains_by_level[lvl.item()]
            n_envs   = group.shape[0]
            n_cands = candidate_indices.shape[0]
            idxs = torch.arange(n_envs, device=self.device) % n_cands
            chosen_rows = candidate_indices[idxs]
            self.bx_start[group] = self.terrain_details[chosen_rows, 10]
            self.bx_end[group]   = self.terrain_details[chosen_rows, 11]
            self.by_start[group] = self.terrain_details[chosen_rows, 12]
            self.by_end[group]   = self.terrain_details[chosen_rows, 13]
            self.compliance[group]  = self.terrain_details[chosen_rows, 6].bool()
            self.system_idx[group]  = self.terrain_details[chosen_rows, 7].long()
            rows = self.terrain_details[chosen_rows, 2].long()
            cols = self.terrain_details[chosen_rows, 3].long()
            self.env_origins[group] = self.terrain_origins[rows, cols]

        # Update compliance and stored PBD parameters for these newly changed envs
        self.set_compliance(env_ids)
        self.store_pbd_params(env_ids) 

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        simulation_context = SimulationContext.instance()
        simulation_context.get_physics_context().enable_gpu_dynamics(True)
        simulation_context.get_physics_context().set_broadphase_type("GPU")
        self.get_terrain()
        self.get_anymal()
        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"])
        self._anymals = A1View(
            prim_paths_expr="/World/envs/.*/a1", name="a1_view", track_contact_forces=True
        )
        if self._particles_active:
            self.create_particle_systems()
            self.particle_system_view = ParticleSystemView(prim_paths_expr="/World/particleSystem/*")
            scene.add(self.particle_system_view)
        scene.add(self._anymals)
        scene.add(self._anymals._thigh)
        scene.add(self._anymals._base)
        scene.add(self._anymals._foot)
        scene.add(self._anymals._calf)


    def initialize_views(self, scene):
        # initialize terrain variables even if we do not need to re-create the terrain mesh
        self.get_terrain(create_mesh=False)

        super().initialize_views(scene)
        if scene.object_exists("a1_view"):
            scene.remove_object("a1_view", registry_only=True)
        if scene.object_exists("thigh_view"):
            scene.remove_object("thigh_view", registry_only=True)
        if scene.object_exists("base_view"):
            scene.remove_object("base_view", registry_only=True)
        if scene.object_exists("foot_view"):
            scene.remove_object("foot_view", registry_only=True)
        if scene.object_exists("calf_view"):
            scene.remove_object("calf_view", registry_only=True)
        if scene.object_exists("particle_system_view"):
            scene.remove_object("particle_system_view", registry_only=True)
        self._anymals = A1View(
            prim_paths_expr="/World/envs/.*/a1", name="a1_view", track_contact_forces=True
        )
        if self._particles_active:
            self.create_particle_systems()
            self.particle_system_view = ParticleSystemView(prim_paths_expr="/World/particleSystem/*")
            scene.add(self.particle_system_view)  
        scene.add(self._anymals)
        scene.add(self._anymals._thigh)
        scene.add(self._anymals._base)
        scene.add(self._anymals._foot)
        scene.add(self._anymals._calf)

    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._create_trimesh(create_mesh=create_mesh)
        levels = self.terrain_details[:, 1].long()  # shape: (N, ) where N=# of terrain blocks
        unique_levels = torch.unique(levels)
        for lvl in unique_levels:
            mask = (levels == lvl)
            row_indices = torch.nonzero(mask, as_tuple=False).flatten()
            self._terrains_by_level[lvl.item()] = row_indices

    def get_anymal(self):
        anymal_translation = torch.tensor([0.0, 0.0, 0.42])
        anymal_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        anymal = A1(
            prim_path=self.default_zero_env_path + "/a1",
            name="a1",
            translation=anymal_translation,
            orientation=anymal_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "a1", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("a1")
        )
        anymal.set_a1_properties(self._stage, anymal.prim)
        anymal.prepare_contacts(self._stage, anymal.prim)

        self.dof_names = anymal.dof_names
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def _randomize_dof_props(self, env_ids=None):
        """Randomize the properties of the DOFs for the given environment IDs."""
        # If no env_ids are provided, operate on all environments
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if self._task_cfg["env"]["randomizationRanges"]["randomizeMotorStrength"]:
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  self.motor_strength_range[1] - self.motor_strength_range[0]) + self.motor_strength_range[0]
        if self._task_cfg["env"]["randomizationRanges"]["randomizeMotorOffset"]:
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     self.motor_offset_range[1] - self.motor_offset_range[0]) + self.motor_offset_range[0]
        if self._task_cfg["env"]["randomizationRanges"]["randomizeKpFactor"]:
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  self.Kp_factor_range[1] - self.Kp_factor_range[0]) + self.Kp_factor_range[0]
        if self._task_cfg["env"]["randomizationRanges"]["randomizeKdFactor"]:
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  self.Kd_factor_range[1] - self.Kd_factor_range[0]) + self.Kd_factor_range[0]
    
    def _randomize_gravity(self):
        
        external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                requires_grad=False) * (self.gravity_range[1] - self.gravity_range[0]) + self.gravity_range[0]
        self.gravities[:, :] = external_force.unsqueeze(0)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        self.world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )

    def _set_mass(self, view, env_ids):
        """Update material properties for a given asset."""

        masses = self.default_base_masses
        distribution_parameters = self.added_mass_range
        set_masses = view.set_masses
        self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (distribution_parameters[1] - distribution_parameters[0]) + distribution_parameters[0]
        masses += self.payloads
        set_masses(masses)
        print(f"Masses updated: {masses}")
        print(f"default_inertia: {self.default_inertias}")
        # Compute the ratios of the new masses to the default masses.
        ratios = masses / self.default_base_masses
        # The default_inertia is scaled by these ratios.
        # Note: The multiplication below assumes broadcasting works correctly for your inertia tensor shape.
        new_inertias = self.default_inertias * ratios.unsqueeze(-1)
        view.set_inertias(new_inertias)
        print(f"Inertias updated: {new_inertias}")

    def _set_friction(self ,asset, env_ids, device="cpu"):
        """Update material properties for a given asset."""
                # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.int64, device=device)
        else:
            env_ids = env_ids.cpu()
        materials = asset._physics_view.get_material_properties().to(device)
        
        print(f"Current materials: {materials}")

        # obtain parameters for sampling friction and restitution values
        static_friction_range = self.friction_range
        dynamic_friction_range = self.friction_range
        restitution_range = self.restitution_range
        num_buckets = 64
        # sample material properties from the given ranges
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device=device)
        material_buckets = torch.rand(*(num_buckets, 3), device=device) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        material_buckets[:, 1] = torch.min(material_buckets[:, 0], material_buckets[:, 1])

        # randomly assign material IDs to the geometries
        shapes_per_env = 4

        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), shapes_per_env), device=device)
        material_samples = material_buckets[bucket_ids]
        print(f"Material samples: {material_samples}")
        print(f"Material samples shape: {material_samples.shape}")
        # material_samples = material_samples.repeat(1, shapes_per_env, 1)
        # print(f"Material samples shape: {material_samples.shape}")
        new_materials = material_samples.view(len(env_ids)*shapes_per_env, 1, 3)
        print(f"New Material samples shape: {new_materials.shape}")
        #update material buffer with new samples
        materials[:] = new_materials

        # apply to simulation
        asset._physics_view.set_material_properties(materials, env_ids)
        print(f"Updated materials: {materials}")
        print(f"Updated materials shape: {materials.shape}")
        self.static_friction_coeffs = material_samples[:, :, 0].clone().to(self.device)  # shape: (num_envs, shapes_per_env)
        self.dynamic_friction_coeffs = material_samples[:, :, 1].clone().to(self.device)  # shape: (num_envs, shapes_per_env)
        self.restitutions = material_samples[:, :, 2].clone().to(self.device)             # shape: (num_envs, shapes_per_env)
        print("Static friction coefficients:", self.static_friction_coeffs)
        print("Dynamic friction coefficients:", self.dynamic_friction_coeffs)
        print("Restitutions:", self.restitutions)

    def _set_coms(self, view, env_ids):
        """Update material properties for a given view."""

        coms, ori = view.get_coms()
        print(f"Current coms: {coms}")

        distribution_parameters = self.com_displacement_range
        self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * ( distribution_parameters[1] - distribution_parameters[0]) + distribution_parameters[0]
        print(f"Displacements: {self.com_displacements.unsqueeze(1)}")
        coms += self.com_displacements.unsqueeze(1)
        set_coms = view.set_coms
        print(f"New coms: {coms}")
        set_coms(coms, ori)
        print(f"Coms updated: {coms}")


    def set_compliance(self, env_ids=None, sync=False):
        """
        Sets compliant-contact stiffness and damping for each env in `env_ids`.
        - If `self.compliance[env]` is True, the values are sampled from the given self.stiffness_range.
        - If `sync` is True, each environment gets one stiffness/damping value applied to all feet.
        - If `sync` is False, each foot is randomized independently.
        - If `self.compliance[env]` is False, both stiffness and damping are set to zero.
        """
        # If no env_ids are provided, operate on all environments
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if len(env_ids) == 0:
            return

        self.default_compliant_k = float(self._task_cfg["env"]["terrain"]["compliant"]["stiffnes"])
        self.default_compliant_c = float(self._task_cfg["env"]["terrain"]["compliant"]["damping"])

        compliance_rand_cfg = self._task_cfg["env"]["randomizationRanges"] \
                                    ["material_randomization"]["compliance"]
        do_randomize = compliance_rand_cfg.get("enabled", False)

        # Separate env_ids by compliance flag
        compliance_true_ids = env_ids[self.compliance[env_ids]]
        compliance_false_ids = env_ids[~self.compliance[env_ids]]

        # For compliant envs, sample random stiffness/damping from the config ranges
        if do_randomize and len(compliance_true_ids) > 0:
            num_compliant = len(compliance_true_ids)
            k_min, k_max = self.stiffness_range
            c_min, c_max = self.damping_multiplier_range
            # Sample stiffness
            if sync:
                # Sample one value per environment, then apply to all feet
                base_k = torch.rand(num_compliant, device=self.device) * (k_max - k_min) + k_min  # (N,)
                stiffness_values = base_k.unsqueeze(1).expand(-1, self.num_feet)  # (N, num_feet)

                base_mult = torch.rand(num_compliant, device=self.device) * (c_max - c_min) + c_min
                damping_mult_values = base_mult.unsqueeze(1).expand(-1, self.num_feet)

                damping_values = stiffness_values * damping_mult_values
            else:
                # Independent random per foot
                stiffness_values = torch.rand(num_compliant, self.num_feet, device=self.device) * (k_max - k_min) + k_min
                damping_mult_values = torch.rand(num_compliant, self.num_feet, device=self.device) * (c_max - c_min) + c_min
                damping_values = stiffness_values * damping_mult_values

            self.stiffness[compliance_true_ids] = stiffness_values
            self.damping[compliance_true_ids] = damping_values         
        
        elif len(compliance_true_ids) > 0:
            # fallback to your defaults
            self.stiffness[compliance_true_ids] = self.default_compliant_k
            self.damping[compliance_true_ids]   = self.default_compliant_c

        # For non-compliant envs, set both stiffness and damping to zero
        if len(compliance_false_ids) > 0:
            self.stiffness[compliance_false_ids] = 0.0
            self.damping[compliance_false_ids] = 0.0

        # Ensure each environment’s material API is created/applied once
        for env in env_ids.tolist():
            for foot in range(self.num_feet):
                i = env * self.num_feet + foot
                if self._material_apis[i] is None:
                    if self._prims[i].HasAPI(PhysxSchema.PhysxMaterialAPI):
                        self._material_apis[i] = PhysxSchema.PhysxMaterialAPI(self._prims[i])
                    else:
                        self._material_apis[i] = PhysxSchema.PhysxMaterialAPI.Apply(self._prims[i])

                # Apply updated stiffness/damping to the material
                k = float(self.stiffness[env, foot])
                self._material_apis[i].CreateCompliantContactStiffnessAttr().Set(k)
                # Only set damping if stiffness > 0 (PhysX auto-disables damping if stiffness is zero)
                if k > 0.0:
                    c = float(self.damping[env, foot])
                    self._material_apis[i].CreateCompliantContactDampingAttr().Set(c)


    def store_pbd_params(self, env_ids=None):
        """
        Populates self.pbd_parameters for each env that has a non-zero system_idx.
        Each row in self.pbd_parameters corresponds to one environment.
        In this example, we store 8 different PBD material parameters per row:
        [friction, damping, viscosity, density, surface_tension, cohesion, adhesion, cfl_coefficient].
        """
        # default to all envs
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if len(env_ids) == 0:
            return

        # Make sure self.pbd_parameters has the right shape:
        # e.g. self.pbd_parameters = torch.zeros((self.num_envs, 8), device=self.device)
        self.pbd_parameters[env_ids] = 0.0  # Reset to 0 for all envs first
        sys_ids      = self.system_idx[env_ids]                       # shape (B,)
        particles_f  = (sys_ids > 0).float()                          # 1.0 if any system
        fluid_list   = [ self._system_is_fluid.get(int(s), 0.0) for s in sys_ids ]
        fluid_f      = torch.tensor(fluid_list, dtype=torch.float32, device=self.device)
        idx_part     = self._pbd_idx["particles_present"]
        idx_fluid    = self._pbd_idx["fluid_present"]
        self.pbd_parameters[env_ids, idx_part]  = particles_f
        self.pbd_parameters[env_ids, idx_fluid] = fluid_f

        # Grab unique systems in the batch
        unique_systems = torch.unique(sys_ids)

        for sid in unique_systems:
            # Skip system_idx <= 0, which we treat as "no system" or invalid system
            if sid <= 0:
                continue
            # Look up the corresponding "systemX" config in _particle_cfg
            sys_id = sid.item()
            system_str = f"system{sys_id}"
            mat_cfg = self._particle_cfg.get(system_str, {})
            mask = (sys_ids == sid)
            target_envs = env_ids[mask]
            if target_envs.numel() == 0:
                continue
            # For each parameter we collected in init_pbd for this system:
            for param_name, _ in self.pbd_by_sys.get(sys_id, []):
                col = self._pbd_idx[param_name]
                val = mat_cfg.get(param_name, 0.0)
                self.pbd_parameters[target_envs, col] = float(val)


    def randomize_pbd_material(self):
        """
        Retrieves an existing PBD material via PhysxSchema and updates its parameters using randomization.
        """
        # Define parameters with their corresponding attribute suffixes once.
        param_to_attr = {
            "pbd_material_friction": "FrictionAttr",
            "pbd_material_density": "DensityAttr",
            "pbd_material_damping": "DampingAttr",
            "pbd_material_particle_friction_scale": "ParticleFrictionScaleAttr",
            "pbd_material_adhesion": "AdhesionAttr",
            "pbd_material_particle_adhesion_scale": "ParticleAdhesionScaleAttr",
            "pbd_material_adhesion_offset_scale": "AdhesionOffsetScaleAttr",
            "pbd_material_viscosity": "ViscosityAttr",
            "pbd_material_surface_tension": "SurfaceTensionAttr",
            "pbd_material_cohesion": "CohesionAttr",
            "pbd_material_lift": "LiftAttr",
            "pbd_material_drag": "DragAttr",
            "pbd_material_cfl_coefficient": "CflCoefficientAttr",
        }

        # Process each system in the configuration.
        for sys_id, param_list in self.pbd_by_sys.items():
            # Skip if the system is not in the config.
            system_name = f"system{sys_id}"
            material_key = f"pbd_material_{system_name}"
            if material_key not in self.created_materials:
                print(f"[WARN] Material {material_key} not found; skipping randomization.")
                continue

            # Get the material API.
            pbd_material_path = f"/World/pbdmaterial_{system_name}"
            material_api = PhysxSchema.PhysxPBDMaterialAPI.Get(self._stage, pbd_material_path)
            if not material_api:
                print(f"[ERROR] Could not find PBD material at {pbd_material_path}")
                continue

            # Pre-sample all parameters once for this system.
            for param_name, (low, high) in param_list:
                sample_val = random.uniform(low, high)
                suffix = param_to_attr.get(param_name)
                if suffix is None:
                    # you could warn here if you like
                    continue

                getter = getattr(material_api, f"Get{suffix}")
                attr   = getter()
                # only set if the attribute actually exists on this material
                if attr and attr.Get() is not None:
                    attr.Set(sample_val)
                    self._particle_cfg[system_name][param_name] = sample_val

            print(f"[INFO] Updated PBD material at {pbd_material_path} with randomized parameters.")
            
    def post_reset(self):
        self.base_init_state = torch.tensor(
            self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            device=self.device,
            requires_grad=False,
        )
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                requires_grad=False)
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_names = ["FL", "FR", "RL", "RR"]
        self.num_feet = len(self.feet_names)
        self.feet_indices = torch.arange(self.num_feet, dtype=torch.long, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)

        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.compliance = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.system_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.bx_start = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.bx_end   = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.by_start = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.by_end   = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        env_ids = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        self.num_dof = self._anymals.num_dof
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.foot_pos = torch.zeros((self.num_envs * 4, 3), dtype=torch.float, device=self.device)
        self.ground_heights_below_foot = torch.zeros((self.num_envs * 4), dtype=torch.float, device=self.device)
        
        self.thigh_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.calf_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_contact_forces = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)


        self.static_friction_coeffs = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.dynamic_friction_coeffs = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                            requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.stiffness = torch.zeros(self.num_envs, self.num_feet, dtype=torch.float, device=self.device, requires_grad=False)
        self.damping = torch.zeros(self.num_envs, self.num_feet, dtype=torch.float, device=self.device, requires_grad=False)

        _prim_paths = find_matching_prim_paths(self._anymals._foot_material_path)
        count = len(_prim_paths)
        self._prims = [get_prim_at_path(path) for path in _prim_paths]
        self._material_apis = [None] * count

        self.friction_scale, self.friction_shift = self.get_scale_shift(self.friction_range)
        self.restitution_scale, self.restitution_shift = self.get_scale_shift(self.restitution_range)
        self.payload_scale, self.payload_shift = self.get_scale_shift(self.added_mass_range)
        self.com_scale, self.com_shift = self.get_scale_shift(self.com_displacement_range)
        self.motor_strength_scale, self.motor_strength_shift = self.get_scale_shift(self.motor_strength_range)
        self.motor_offset_scale, self.motor_offset_shift = self.get_scale_shift(self.motor_offset_range)
        self.Kp_factor_scale, self.Kp_factor_shift = self.get_scale_shift(self.Kp_factor_range)
        self.Kd_factor_scale, self.Kd_factor_shift = self.get_scale_shift(self.Kd_factor_range)
        self.stiffness_scale, self.stiffness_shift = self.get_scale_shift(self.stiffness_range)
        self.damping_scale, self.damping_shift = self.get_scale_shift(self.damping_range)
        self.gravity_scale, self.gravity_shift = self.get_scale_shift(self.gravity_range)

        self.default_base_masses = self._anymals._base.get_masses().clone()
        self.default_inertias = self._anymals._base.get_inertias().clone()
        # self.default_materials = self._anymals._foot._physics_view.get_material_properties().to(self.device)
        body_masses = self._anymals.get_body_masses().clone()  # already a torch tensor
        self.total_masses = torch.sum(body_masses, dim=1).to(self.device)

        # Determine the highest terrain level from the terrain details.
        self.highest_level = int(self.terrain_details[:, 1].max().item())

        # Get joint limits
        dof_limits = self._anymals.get_dof_limits()
        lower_limits = dof_limits[0, :, 0]    
        upper_limits = dof_limits[0, :, 1]    
        midpoint = 0.5 * (lower_limits + upper_limits)
        limit_range = upper_limits - lower_limits
        soft_lower_limits = midpoint - 0.5 * limit_range * self.soft_dof_pos_limit
        soft_upper_limits = midpoint + 0.5 * limit_range * self.soft_dof_pos_limit
        self.a1_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.a1_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.a1_dof_soft_lower_limits = soft_lower_limits.to(device=self._device)
        self.a1_dof_soft_upper_limits = soft_upper_limits.to(device=self._device)

        self.dof_vel_limits = self._anymals._physics_view.get_dof_max_velocities()[0].to(device=self._device)
        self.torque_limits = self._anymals._physics_view.get_dof_max_forces()[0].to(device=self._device)

        if self._task_cfg["env"]["randomizationRanges"]["randomizeAddedMass"]:
            self._set_mass(self._anymals._base, env_ids=env_ids)
        if self._task_cfg["env"]["randomizationRanges"]["randomizeCOM"]:
            self._set_coms(self._anymals._base, env_ids=env_ids)
        if self._task_cfg["env"]["randomizationRanges"]["randomizeFriction"]:
            self._set_friction(self._anymals._foot, env_ids=env_ids)


        self._prepare_reward_function()

        # Define maximum length of tracking history
        self.tracking_history_len = 10
        # Initialize linear velocity tracking buffer and index for each env
        self.tracking_lin_vel_x_history = torch.zeros((self.num_envs, self.tracking_history_len), device=self.device)
        self.tracking_lin_vel_x_history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tracking_lin_vel_x_history_full = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Initialize angular velocity tracking buffer and index for each env
        self.tracking_ang_vel_x_history = torch.zeros((self.num_envs, self.tracking_history_len), device=self.device)
        self.tracking_ang_vel_x_history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tracking_ang_vel_x_history_full = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Initialize episode-length tracking buffer
        self.ep_length_history = torch.zeros((self.num_envs, self.tracking_history_len), device=self.device)
        self.ep_length_history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ep_length_history_full = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.reset_idx(env_ids)
        self.init_done = True

        
    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        if self.vel_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        if self.curriculum:
            self.update_terrain_level(env_ids)

        self._randomize_dof_props(env_ids)

        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        if self.curriculum:
            jitter_x = jitter_y = 1.0
        else:
            jitter_x = self.terrain.env_rows * self.terrain.env_length 
            jitter_y = self.terrain.env_cols * self.terrain.env_width 
            
        # generate an N×1 tensor, then squeeze to (N,)
        rand_x = torch_rand_float(
            0, jitter_x,
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        rand_y = torch_rand_float(
            0, jitter_y,
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)

        self.base_pos[env_ids, 0] += rand_x
        self.base_pos[env_ids, 1] += rand_y

        rand_yaw = torch_rand_float(0, 2 * np.pi, (len(env_ids), 1), device=self.device)
        random_quat = torch.cat([
            torch.cos(rand_yaw / 2),
            torch.zeros(len(env_ids), 2, device=self.device),
            torch.sin(rand_yaw / 2)
        ], dim=1)
        self.base_quat[env_ids] = random_quat        
        self.base_velocities[env_ids] = self.base_init_state[7:]

        self._anymals.set_world_poses(
            positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices
        )
        self._anymals.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._anymals.set_joint_positions(positions=self.dof_pos[env_ids].clone(), indices=indices)
        self._anymals.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)

        # Print environment origins and levels for each reset
        for env in env_ids:
            origin = self.env_origins[env].cpu().tolist()
            level = int(self.terrain_levels[env].item())
            print(f"[Reset] Env {env.item()} - Origin: {origin}, Level: {level}")

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(
            1
        )  # set small commands to zero

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )

        # Get a per‐environment reward in X, instead of a single scalar
        lin_x_rewards = self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s  # shape == (len(env_ids),)
        ang_x_rewards = self.episode_sums["tracking_ang_vel"][env_ids] / self.max_episode_length_s  # shape == (len(env_ids),)
        final_lengths = self.progress_buf[env_ids]  / self.max_episode_length_s     # Update history buffer for each environment

        # 1. current write positions for every env we’re resetting
        lin_pos = self.tracking_lin_vel_x_history_idx[env_ids] % self.tracking_history_len
        ang_pos = self.tracking_ang_vel_x_history_idx[env_ids] % self.tracking_history_len
        len_pos = self.ep_length_history_idx[env_ids]      % self.tracking_history_len   # episode length

        # 2. write the new values
        self.tracking_lin_vel_x_history[env_ids, lin_pos] = lin_x_rewards                 # (N,)
        self.tracking_ang_vel_x_history[env_ids, ang_pos] = ang_x_rewards                 # (N,)
        self.ep_length_history      [env_ids, len_pos] = final_lengths                    # (N,)

        # 3. advance the circular indices (modulo history length)
        self.tracking_lin_vel_x_history_idx[env_ids] = (lin_pos + 1) % self.tracking_history_len
        self.tracking_ang_vel_x_history_idx[env_ids] = (ang_pos + 1) % self.tracking_history_len
        self.ep_length_history_idx      [env_ids] = (len_pos + 1) % self.tracking_history_len

        # 4. mark a buffer as ‘full’ the first time it wraps around
        self.tracking_lin_vel_x_history_full[env_ids] |= (self.tracking_lin_vel_x_history_idx[env_ids] == 0)
        self.tracking_ang_vel_x_history_full[env_ids] |= (self.tracking_ang_vel_x_history_idx[env_ids] == 0)
        self.ep_length_history_full      [env_ids] |= (self.ep_length_history_idx      [env_ids] == 0)

        for key in self.episode_sums.keys():      
            self.episode_sums[key][env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.   

        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        self.extras["time_outs"] = self.timeout_buf


    def refresh_dof_state_tensors(self):
        self.dof_pos = self._anymals.get_joint_positions(clone=False)
        self.dof_vel = self._anymals.get_joint_velocities(clone=False)

    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._anymals.get_world_poses(clone=False)
        self.base_velocities = self._anymals.get_velocities(clone=False)
        self.foot_pos, _ = self._anymals._foot.get_world_poses(clone=False)

    def refresh_net_contact_force_tensors(self):
        # self.foot_contact_forces = self.foot_contact_forces * 0.9 + self._anymals._foot.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3) * 0.1
        self.foot_contact_forces = self._anymals._foot.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3)
        self.thigh_contact_forces = self._anymals._thigh.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3)
        self.calf_contact_forces = self._anymals._calf.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 4, 3)
        self.base_contact_forces = self._anymals._base.get_net_contact_forces(dt=self.dt,clone=False).view(self._num_envs, 3)

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return

        self.actions = actions.clone().to(self.device)

        for i in range(self.decimation):
            if self.world.is_playing():
                self.joint_pos_target = self.action_scale * self.actions + self.default_dof_pos
                torques = self.Kp * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.Kd * self.Kd_factors * self.dof_vel
                torques = torques * self.motor_strengths
                max_tau = self.torque_limits
                torques = torch.clip(torques, -max_tau, max_tau)
                self._anymals.set_joint_efforts(torques)
                self.torques = torques
                SimulationContext.step(self.world, render=False)
                self.refresh_dof_state_tensors()

    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self.world.is_playing():

            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()
            self.refresh_net_contact_force_tensors()

            self.common_step_counter += 1
            if self.push_enabled and (self.common_step_counter % self.push_interval) == 0:
                self.push_robots()
            if  self.randomize_gravity and (self.common_step_counter % self.gravity_randomize_interval) == 0:
                self._randomize_gravity()
            if self.randomize_pbd and (self.common_step_counter % self.randomize_pbd_interval == 0):
                self.randomize_pbd_material()

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

            if self.measure_heights:
                if self._particles_active:
                    self.query_top_particle_positions(visualize=True) 
                self.get_heights_below_foot()

            self.check_termination()
            self.compute_reward()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]
            
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self):
        self.base_velocities[:, 0:2] = torch_rand_float(
            -1.0, 1.0, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self._anymals.set_velocities(self.base_velocities)

    def check_termination(self):
        self.timeout_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        )
        self.has_fallen = (torch.norm(self.base_contact_forces, dim=1) > 1.0) 
        self.reset_buf = self.has_fallen.clone()
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)
        

        if self.teleport_active:
            # Convert each robot's base (x,y) position into heightfield indices
            hf_x = (self.base_pos[:, 0] + self.terrain.border_size) / self.terrain.horizontal_scale
            hf_y = (self.base_pos[:, 1] + self.terrain.border_size) / self.terrain.horizontal_scale
            # Define a small distance buffer for early reset
            # Check if the robot is outside the "safe" bounds with the buffer:
            out_of_bounds = (
                (hf_x < self.bx_start + self.teleport_buffer) |
                (hf_x > self.bx_end - self.teleport_buffer)  |
                (hf_y < self.by_start + self.teleport_buffer) |
                (hf_y > self.by_end - self.teleport_buffer)
            )

            # Instead of marking them to reset, teleport them back to spawn:
            teleport_env_ids = out_of_bounds.nonzero(as_tuple=False).flatten()
            if teleport_env_ids.numel() > 0:
                self.base_pos[teleport_env_ids] = self.base_init_state[0:3]
                self.base_pos[teleport_env_ids, 0:3] += self.env_origins[teleport_env_ids]
                self.base_pos[teleport_env_ids, 0:2] += torch_rand_float(-1., 1., (len(teleport_env_ids), 2), device=self.device)
                indices = teleport_env_ids.to(dtype=torch.int32)
                self._anymals.set_world_poses(
                positions=self.base_pos[teleport_env_ids].clone(), orientations=self.base_quat[teleport_env_ids].clone(), indices=indices
            )
            
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """

        # 1) Discover all reward methods
        self.all_reward_methods = [name[8:] for name in dir(self) if name.startswith('_reward_')]

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        self.inactive_reward_names = [
            name
            for name in self.all_reward_methods
            if name not in self.reward_scales
        ]

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # 5) Initialize episode_sums _dict_ before populating it
        self.episode_sums = {}
        # active rewards
        for name in self.reward_names:
            self.episode_sums[name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )

        # inactive rewards, prefixed
        for name in self.inactive_reward_names:
            self.episode_sums[f"inactive_{name}"] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
            )

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        # 1) log inactive rewards under “inactive_<name>”
        for name in self.inactive_reward_names:
            raw = getattr(self, f"_reward_{name}")()
            # e.g. store in extras or episode_sums:
            self.episode_sums[f"inactive_{name}"] = raw * self.dt

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew



    def get_observations(self):
        """
        Build a 'proprio_obs' block and (optionally) a 'heights' block,
        then combine them in obs_buf. Only 'proprio_obs' is put into obs_history.
        """
        # 1) Collect all your normal (proprio) observations:
        proprio_obs = torch.cat((
            self.base_lin_vel * self.lin_vel_scale,
            self.base_ang_vel * self.ang_vel_scale,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            self.dof_pos * self.dof_pos_scale,
            self.dof_vel * self.dof_vel_scale,
            torch.norm(self.foot_contact_forces[:, self.feet_indices, :], dim=-1),
            self.actions,
        ), dim=-1)  # this should match self._num_proprio in size

        # 2) Add noise (only on proprio)
        if self.add_noise:
            proprio_obs += (2.0 * torch.rand_like(proprio_obs) - 1.0) * self.noise_scale_vec
        proprio_obs = torch.clip(proprio_obs, -self.clip_obs, self.clip_obs)

        # 3) If measuring heights, compute them and concatenate AFTER the proprio block.
        if self.measure_heights:
            if self.debug_heights:
                self._visualize_height_scans()

            heights = torch.clip(
                self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1.0, 1.0
            ) * self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * self._task_cfg["env"]["learn"]["noiseLevel"] * self.height_meas_scale
            
            final_obs_no_history = torch.cat([proprio_obs, heights], dim=-1)
        else:
            final_obs_no_history = proprio_obs
        # 4) Build privileged observations
        priv_parts = []
        if self.priv_base:
            priv_parts += [
                (self.static_friction_coeffs - self.friction_shift) * self.friction_scale,
                (self.dynamic_friction_coeffs - self.friction_shift) * self.friction_scale,
                (self.restitutions - self.restitution_shift) * self.restitution_scale,
                (self.payloads.unsqueeze(1) - self.payload_shift) * self.payload_scale,
                (self.com_displacements - self.com_shift) * self.com_scale,
                (self.motor_strengths - self.motor_strength_shift) * self.motor_strength_scale,
            ]

        # compliance (soft contacts)
        if self.priv_compliance and self._compliance_active:
            priv_parts += [
                (self.stiffness - self.stiffness_shift) * self.stiffness_scale,
                (self.damping - self.damping_shift) * self.damping_scale,
            ]

        # PBD particle parameters
        if self._particles_active and self.priv_pbd_particle:
            priv_parts.append((self.pbd_parameters - self.pbd_shift) * self.pbd_scale)

        # finally concatenate all privileged parts (or an empty tensor if none)
        if priv_parts:
            priv_buf = torch.cat(priv_parts, dim=1)
        else:
            # zero-length tensor per env
            priv_buf = torch.zeros((self.num_envs, 0), device=self.device, dtype=torch.float)

        # 5) Concatenate everything: [ (proprio + maybe heights) + priv_buf + obs_history ]
        self.obs_buf = torch.cat([
            final_obs_no_history,
            priv_buf,
            self.obs_history_buf.view(self.num_envs, -1)
        ], dim=-1)

        self.obs_history_buf = torch.where(
            (self.progress_buf <= 1)[:, None, None],                     # On (re)reset
            torch.stack([final_obs_no_history] * self._obs_history_length,   # fill history
                        dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],           # drop oldest frame
                final_obs_no_history.unsqueeze(1)          # append newest
            ], dim=1)
        )

    def get_heights_below_foot(self, env_ids=None):

        foot_positions = self.foot_pos.view(self.num_envs, 4, 3)
        if env_ids is not None:
            foot_positions  = foot_positions[env_ids]      
        N = foot_positions.shape[0]
        points = (foot_positions.unsqueeze(2) + self.height_points.view(N, 4, -1, 3)).reshape(N, self._num_height_points, 3)

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        self.height_px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        self.height_py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[self.height_px, self.height_py]
        heights2 = self.height_samples[self.height_px+1, self.height_py]
        heights3 = self.height_samples[self.height_px, self.height_py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        self.measured_heights = heights.view(self.num_envs, -1) * self.terrain.vertical_scale
 
    def get_scale_shift(self, rng):
        rng_tensor = torch.tensor(rng, dtype=torch.float, device=self.device)
        # Check if the range is degenerate (both elements are the same)
        if rng_tensor[1] == rng_tensor[0]:
            scale = 0.0
            shift = rng_tensor[0]  # or you could use rng_tensor[1] since they're equal
        else:
            scale = 2.0 / (rng_tensor[1] - rng_tensor[0])
            shift = (rng_tensor[1] + rng_tensor[0]) / 2.0
        return scale, shift
    
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        thigh_contact = (
            torch.norm(self.thigh_contact_forces, dim=-1)
            > 0.1
        )
        calf_contact = (torch.norm(self.calf_contact_forces, dim=-1) > 0.1)
        total_contact = thigh_contact + calf_contact
        return torch.sum(total_contact, dim=-1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.has_fallen
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.a1_dof_soft_lower_limits).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.a1_dof_soft_upper_limits).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.only_positive_rewards).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.foot_contact_forces[:, self.feet_indices, 2] > 1.0  # Placeholder for contact detection, adjust threshold as needed
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt  # Assuming self.dt is the timestep duration
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.foot_contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.foot_contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.foot_contact_forces[:, self.feet_indices, :], dim=-1) -  self.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_hip_motion(self):
        # Penalize hip motion
        return torch.sum(torch.abs(self.dof_pos[:, :4] - self.default_dof_pos[:, :4]), dim=1)
    
    #------------ end reward functions----------------
    


    #------------ particle based functions----------------
    def create_particle_systems(self):
        for i in range(self.terrain_details.shape[0]):
            terrain_row = self.terrain_details[i]
            if not int(terrain_row[5].item()):
                continue  # Skip terrains without particles

            # Construct the system name from integer system_id
            system_id = int(terrain_row[7])                # e.g. 1, 2, ...
            system_name = f"system{system_id}"     # "system1", "system2", etc.            
            material_key = f"pbd_material_{system_name}"
            particle_system_path = f"/World/particleSystem/{system_name}"
            contact_offset = self._particle_cfg[system_name].get("particle_system_contact_offset", None)

            # **Create Particle System if not already created**
            if system_name not in self.created_particle_systems:
                if not self._stage.GetPrimAtPath(particle_system_path).IsValid():
                    fluid = self._particle_cfg[system_name].get("particle_grid_fluid", False)
                    particle_contact_offset = self._particle_cfg[system_name].get("particle_system_particle_contact_offset", None)
                    fluid_rest_offset = self._particle_cfg[system_name].get("particle_system_fluid_rest_offset", None)
                    solid_rest_offset = self._particle_cfg[system_name].get("particle_system_solid_rest_offset", None)
                    rest_offset = self._particle_cfg[system_name].get("particle_system_rest_offset", None)

                    # Raise an error if it is a fluid system and no particle contact offset is provided.
                    if fluid and particle_contact_offset is None:
                        raise ValueError("For fluid systems, 'particle_system_particle_contact_offset' must be provided.")

                    if fluid:
                        if fluid_rest_offset is None :
                            fluid_rest_offset = 0.99 * 0.6 * particle_contact_offset

                        # For example, for rest offset and solid rest offset, if they are not provided:
                        if rest_offset is None :
                            rest_offset = 0.99 * particle_contact_offset

                        if solid_rest_offset is None :
                            solid_rest_offset = 0.99 * particle_contact_offset

                        if contact_offset is None :
                            contact_offset = 1 * particle_contact_offset

                    particle_system = ParticleSystem(
                        prim_path=particle_system_path,
                        particle_system_enabled=True,
                        simulation_owner="/physicsScene",
                        rest_offset=rest_offset,
                        contact_offset=contact_offset,
                        solid_rest_offset=solid_rest_offset,
                        fluid_rest_offset = fluid_rest_offset, 
                        particle_contact_offset=particle_contact_offset,
                        max_velocity=self._particle_cfg[system_name].get("particle_system_max_velocity", None),
                        max_neighborhood=self._particle_cfg[system_name].get("particle_system_max_neighborhood", None),
                        solver_position_iteration_count=self._particle_cfg[system_name].get("particle_system_solver_position_iteration_count", None),
                        enable_ccd=self._particle_cfg[system_name].get("particle_system_enable_ccd", None),
                        max_depenetration_velocity=self._particle_cfg[system_name].get("particle_system_max_depenetration_velocity", None),
                    )
                    if self._particle_cfg[system_name].get("Anisotropy", False):
                        # apply api and use all defaults
                        PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.prim)

                    if self._particle_cfg[system_name].get("Smoothing", False):
                        # apply api and use all defaults
                        PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.prim)

                    if self._particle_cfg[system_name].get("Isosurface", False):
                        # apply api and use all defaults
                        PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.prim)
                        # tweak anisotropy min, max, and scale to work better with isosurface:
                        if self._particle_cfg[system_name].get("Anisotropy", False):
                            ani_api = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.prim)
                            ani_api.CreateScaleAttr().Set(5.0)
                            ani_api.CreateMinAttr().Set(1.0)  # avoids gaps in surface
                            ani_api.CreateMaxAttr().Set(2.0)  # avoids gaps in surface

                    print(f"[INFO] Created Particle System: {particle_system_path}")
                self.created_particle_systems[system_name] = particle_system_path

            # **Create PBD Material if not already created**
            if material_key not in self.created_materials:
                self.create_pbd_material(system_name)
                self.created_materials[material_key] = True

            # **Create Particle Grid under the existing system**
            self.create_particle_grid(i, terrain_row, system_name)
        print(f"[INFO] Created {len(self.created_materials)} PBD Materials.")
        print(f"[INFO] Created {self.total_particles} Particles.")


    def create_pbd_material(self, system_name):
        # Retrieve material parameters from config based on system_name
        material_cfg = self._particle_cfg[system_name]
        
        # Define unique material path
        pbd_material_path = f"/World/pbdmaterial_{system_name}"
        
        # Check if the material already exists
        if not self._stage.GetPrimAtPath(pbd_material_path).IsValid():
            # Create PBD Material
            particleUtils.add_pbd_particle_material(
                self._stage,
                Sdf.Path(pbd_material_path),
                friction=material_cfg.get("pbd_material_friction", None),
                particle_friction_scale=material_cfg.get("pbd_material_particle_friction_scale", None),
                damping=material_cfg.get("pbd_material_damping", None),
                viscosity=material_cfg.get("pbd_material_viscosity", None),
                vorticity_confinement=material_cfg.get("pbd_material_vorticity_confinement", None),
                surface_tension=material_cfg.get("pbd_material_surface_tension", None),
                cohesion=material_cfg.get("pbd_material_cohesion", None),
                adhesion=material_cfg.get("pbd_material_adhesion", None),
                particle_adhesion_scale=material_cfg.get("pbd_material_particle_adhesion_scale", None),
                adhesion_offset_scale=material_cfg.get("pbd_material_adhesion_offset_scale", None),
                gravity_scale=material_cfg.get("pbd_material_gravity_scale", None),
                lift=material_cfg.get("pbd_material_lift", None),
                drag=material_cfg.get("pbd_material_drag", None),
                density=material_cfg.get("pbd_material_density", None),
                cfl_coefficient=material_cfg.get("pbd_material_cfl_coefficient", None)
            )
            print(f"[INFO] Created PBD Material: {pbd_material_path}")

            # Assign material to particle system
            ps = PhysxSchema.PhysxParticleSystem.Get(self._stage, Sdf.Path(f"/World/particleSystem/{system_name}"))
            physicsUtils.add_physics_material_to_prim(self._stage, ps.GetPrim(), pbd_material_path)

            if self._particle_cfg[system_name].get("Looks", False):
                mtl_created = []
                omni.kit.commands.execute(
                    "CreateAndBindMdlMaterialFromLibrary",
                    mdl_name="OmniSurfacePresets.mdl",
                    mtl_name="OmniSurface_DeepWater",
                    mtl_created_list=mtl_created,
                    select_new_prim=False,
                )
                material_path = mtl_created[0]
                omni.kit.commands.execute(
                    "BindMaterial", prim_path=Sdf.Path(f"/World/particleSystem/{system_name}"), material_path=material_path
                )


    def create_particle_grid(self, i, terrain_row, system_name):
        # Define the particle system path
        particle_system_path = f"/World/particleSystem/{system_name}"    

        # Extract parameters from terrain_detail and config
        level = int(terrain_row[1])
        row_idx = int(terrain_row[2])
        col_idx = int(terrain_row[3])
        depth = float(terrain_row[8])
        size  = float(terrain_row[9])
    
        # If your environment origins are stored separately:
        env_origin = self.terrain_origins[row_idx, col_idx].float()
        env_origin_x = float(env_origin[0])
        env_origin_y = float(env_origin[1])
        env_origin_z = float(env_origin[2])
        
        x_position = env_origin_x - size / 2.0
        y_position = env_origin_y - size / 2.0
        z_position = env_origin_z + 0.05  # Align with environment origin
        lower = Gf.Vec3f(x_position, y_position, z_position)

        system_cfg = self._particle_cfg[system_name]
        solid_rest_offset = system_cfg.get("particle_system_solid_rest_offset", None)
        particle_spacing_factor = system_cfg.get("particle_grid_spacing", None)
        fluid = system_cfg.get("particle_grid_fluid", None)

        if fluid:
            fluid_rest_offset = 0.99 * 0.6 * system_cfg.get("particle_system_particle_contact_offset", None)
            particle_spacing = particle_spacing_factor * fluid_rest_offset
        else:
            particle_spacing = particle_spacing_factor * solid_rest_offset

        num_samples_x = int(size / particle_spacing) + 1
        num_samples_y = int(size / particle_spacing) + 1
        num_samples_z = int(depth / particle_spacing) + 1

        jitter_factor = system_cfg["particle_grid_jitter_factor"] * particle_spacing

        positions = []
        velocities = []
        uniform_particle_velocity = Gf.Vec3f(0.0)
        ind = 0
        x = lower[0]
        y = lower[1]
        z = lower[2]
        for i in range(num_samples_x):
            for j in range(num_samples_y):
                for k in range(num_samples_z):
                    jitter_x = random.uniform(-jitter_factor, jitter_factor)
                    jitter_y = random.uniform(-jitter_factor, jitter_factor)
                    jitter_z = random.uniform(-jitter_factor, jitter_factor)

                    # Apply jitter to the position
                    jittered_x = x + jitter_x
                    jittered_y = y + jitter_y
                    jittered_z = z + jitter_z
                    positions.append(Gf.Vec3f(jittered_x, jittered_y, jittered_z))
                    velocities.append(uniform_particle_velocity)
                    ind += 1
                    z += particle_spacing
                z = lower[2]
                y += particle_spacing
            y = lower[1]
            x += particle_spacing

        # Define particle point instancer path (now grouped by level)
        particle_point_instancer_path = f"/World/particleSystem/{system_name}/level_{level}/particleInstancer"
        self.particle_instancers_by_level.setdefault(level, [])

        # Check if the PointInstancer already exists to prevent duplication
        if not self._stage.GetPrimAtPath(particle_point_instancer_path).IsValid():
            # Add the particle set to the point instancer
            particleUtils.add_physx_particleset_pointinstancer(
                self._stage,
                Sdf.Path(particle_point_instancer_path),
                Vt.Vec3fArray(positions),
                Vt.Vec3fArray(velocities),
                Sdf.Path(particle_system_path),
                self._particle_cfg[system_name]["particle_grid_self_collision"],
                self._particle_cfg[system_name]["particle_grid_fluid"],
                self._particle_cfg[system_name]["particle_grid_particle_group"],
                self._particle_cfg[system_name]["particle_grid_particle_mass"],
                self._particle_cfg[system_name]["particle_grid_density"],
                num_prototypes=1,  # Adjust if needed
                prototype_indices=None  # Adjust if needed
            )
            print(f"[INFO] Created Particle Grid at {particle_point_instancer_path}")
            self.particle_instancers_by_level[level].append(particle_point_instancer_path)
            # Increment the total_particles counter
            self.total_particles += len(positions)
        
            # Configure particle prototype
            particle_prototype_sphere = UsdGeom.Sphere.Get(
                self._stage, Sdf.Path(particle_point_instancer_path).AppendChild("particlePrototype0")
            )
            if fluid:
                radius = fluid_rest_offset 
            else:
                radius = solid_rest_offset
            particle_prototype_sphere.CreateRadiusAttr().Set(radius)
            # Increase counters, etc.
            self.total_particles += len(positions)
            print(f"[INFO] Created {len(positions)} Particles at {particle_point_instancer_path}")
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
            # Increment the total_particles counter
            self.total_particles += len(new_positions)

    def query_top_particle_positions(self, visualize=False):
        """
        Query all particle positions from the given level and, using the depression indices
        for that level, find for each grid cell the
        top (maximum z) particle position.

        Returns:
            A dictionary mapping cell indices (i, j) to a tuple:
            (cell_center_x, cell_center_y, top_z)
            Only cells where at least one particle was found are included.
        """
        if not self.particle_instancers_by_level:
            print("No particle instancers registered yet; skipping top particle query.")
            return
        stage = self._stage
        # Determine which levels to process: either the given level or all levels present
        levels_to_process = list(self.particle_instancers_by_level.keys())
        positions = []
        proto_indices = []
        # Iterate each level and its systems
        for lvl in levels_to_process:
            env_ids = (self.terrain_levels == lvl).nonzero(as_tuple=False).flatten()
            if env_ids.numel() == 0:
                # No environments at this level, skip
                continue
            # gather all particle positions for this level
            all_positions = []
            for inst_path in self.particle_instancers_by_level[lvl]:
                prim = stage.GetPrimAtPath(inst_path)
                if not prim.IsValid():
                    continue
                position = UsdGeom.PointInstancer(prim).GetPositionsAttr().Get() or Vt.Vec3fArray()
                if position:
                    all_positions.append(np.array(position))

            if not all_positions:
                continue

            particle_positions_np = np.vstack(all_positions)


            foot_positions = self.foot_pos.view(self.num_envs, 4, 3)  
            if env_ids is not None:
                foot_positions  = foot_positions[env_ids]    
            N = foot_positions.shape[0]
            points = (foot_positions.unsqueeze(2) + self.particle_height_points.view(N, 4, -1, 3)).reshape(N, self._num_particle_height_points, 3)
            points += self.terrain.border_size
            points = (points / self.terrain.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            grid_indices = np.stack([px.cpu().numpy(), py.cpu().numpy()], axis=1)
            unique_grid_indices = np.unique(grid_indices, axis=0)
            # Prepare cell boundaries in vectorized form
            cell_scale = self.terrain.horizontal_scale
            border_size = self.terrain.border_size

            cell_x = unique_grid_indices[:, 0] * cell_scale - border_size
            cell_y = unique_grid_indices[:, 1] * cell_scale - border_size
            half_scale = cell_scale / 2

            cell_x_min = cell_x - half_scale
            cell_x_max = cell_x + half_scale
            cell_y_min = cell_y - half_scale
            cell_y_max = cell_y + half_scale

            # Vectorized mask computation
            particle_x = particle_positions_np[:, 0][:, None]
            particle_y = particle_positions_np[:, 1][:, None]

            mask = (
                (particle_x >= cell_x_min) &
                (particle_x < cell_x_max) &
                (particle_y >= cell_y_min) &
                (particle_y < cell_y_max)
            )

            # Compute top Z in a vectorized manner
            for idx, (i, j) in enumerate(unique_grid_indices):
                particle_mask = mask[:, idx]
                if particle_mask.any():
                    top_z = np.min([particle_positions_np[particle_mask, 2].max(), 0])
                    self.height_samples[i, j] = int(round(top_z / self.terrain.vertical_scale))

                if visualize:
                    positions.append(Gf.Vec3f(cell_x[idx], cell_y[idx], float(self.height_samples[i, j])))
                    proto_indices.append(0)

        if visualize:
            positions_array = Vt.Vec3fArray(positions)
            proto_indices_array = Vt.IntArray(proto_indices)

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

            # 5) Assign to the PointInstancer
            point_instancer.CreatePositionsAttr().Set(positions_array)
            point_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)

            # (Optional) Color the debug spheres differently
            sphere_geom = UsdGeom.Sphere(stage.GetPrimAtPath(sphere_proto_path))
            sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])  # green for clarity


    def _init_height_scan_instancer(self):
        """
        Visualizes the height-scan points more efficiently by using a single PointInstancer
        to display all the debug spheres instead of creating/updating individual prims.
        """
        if not self.world.is_playing():
            return

        # 1) Create/Get a dedicated DebugHeight scope
        parent_scope_path = "/World/DebugHeight"
        parent_scope_prim = self._stage.GetPrimAtPath(parent_scope_path)
        if not parent_scope_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Scope",
                prim_path=parent_scope_path,
                attributes={}
            )

        # 2) Create/Get a single PointInstancer for all height scan debug spheres
        point_instancer_path = f"{parent_scope_path}/HeightScanPointInstancer"
        point_instancer_prim = self._stage.GetPrimAtPath(point_instancer_path)
        if not point_instancer_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="PointInstancer",
                prim_path=point_instancer_path,
                attributes={}
            )
        self._height_scan_instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(point_instancer_path))

        # 3) Create/ensure a single prototype sphere (with a small radius)
        prototype_path = f"{point_instancer_path}/prototype_Sphere"
        prototype_prim = self._stage.GetPrimAtPath(prototype_path)
        if not prototype_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrim",
                prim_type="Sphere",
                prim_path=prototype_path,
                attributes={"radius": 0.02},
            )
        # Make sure the PointInstancer references the prototype
        if len(self._height_scan_instancer.GetPrototypesRel().GetTargets()) == 0:
            self._height_scan_instancer.GetPrototypesRel().AddTarget(prototype_path)
    
        # 6) Set a debug color on the prototype sphere
        sphere_geom = UsdGeom.Sphere(self._stage.GetPrimAtPath(prototype_path))
        sphere_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        
    def _visualize_height_scans(self):
        # lazy init
        if not hasattr(self, "_height_scan_instancer"):
            self._init_height_scan_instancer()

        px      = self.height_px.flatten()
        py      = self.height_py.flatten()
        pz = self.measured_heights.flatten()
        # compute world coords in bulk
        hscale = self.terrain.horizontal_scale
        bsize  = self.terrain.border_size
        xs = px * hscale - bsize
        ys = py * hscale - bsize
        # build one Vec3fArray
        vecs = [Gf.Vec3f(float(x), float(y), float(z))
           for x, y, z in zip(xs, ys, pz)]
        positions_array = Vt.Vec3fArray(vecs)
        # proto indices all zero
        proto_indices_array = Vt.IntArray([0] * len(vecs))

        # 5) Update the PointInstancer with the positions and prototype indices
        self._height_scan_instancer.CreatePositionsAttr().Set(positions_array)
        self._height_scan_instancer.CreateProtoIndicesAttr().Set(proto_indices_array)



#------------ helper functions----------------

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))
