


import asyncio
from abc import abstractmethod

import numpy as np
import omni.kit
import omni.usd
import torch
from gym import spaces
from omni.isaac.cloner import GridCloner
from .base_task import BaseTask
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from .rl_task_interface import RLTaskInterface
from pxr import Gf, UsdGeom, UsdLux


class RLTask(RLTaskInterface):

    """This class provides a PyTorch RL-specific interface for setting up RL tasks.
    It includes utilities for setting up RL task related parameters,
    cloning environments, and data collection for RL algorithms.
    """

    def __init__(self, name, env, offset=None) -> None:

        """Initializes RL parameters, cloner object, and buffers.

        Args:
            name (str): name of the task.
            env (VecEnvBase): an instance of the environment wrapper class to register task.
            offset (Optional[np.ndarray], optional): offset applied to all assets of the task. Defaults to None.
        """

        BaseTask.__init__(self, name=name, offset=offset)

        self._rand_seed = self._cfg["seed"]
        # optimization flags for pytorch JIT
        torch._C._jit_set_nvfuser_enabled(False)

        self.test = self._cfg["test"]
        self._device = self._cfg["sim_device"]

        # set up replicator for camera data collection
        self.enable_cameras = self._task_cfg["sim"].get("enable_cameras", False)
        if self.enable_cameras:
            from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
            from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener
            import omni.replicator.core as rep

            self.rep = rep
            self.PytorchWriter = PytorchWriter
            self.PytorchListener = PytorchListener

        print("Task Device:", self._device)

        self.clip_obs = self._task_cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self._task_cfg["env"].get("clipActions", np.Inf)
        self.rl_device = self._cfg.get("rl_device", "cuda:0")

        self.control_frequency_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        self.rendering_interval = self._task_cfg.get("renderingInterval", 1)

        # parse default viewport camera position and lookat target and resolution (width, height)
        self.camera_position = [10, 10, 3]
        self.camera_target = [0, 0, 0]
        self.viewport_camera_width = 1280
        self.viewport_camera_height = 720
        if "viewport" in self._task_cfg:
            self.camera_position = self._task_cfg["viewport"].get("camera_position", self.camera_position)
            self.camera_target = self._task_cfg["viewport"].get("camera_target", self.camera_target)
            self.viewport_camera_width = self._task_cfg["viewport"].get("viewport_camera_width", self.viewport_camera_width)
            self.viewport_camera_height = self._task_cfg["viewport"].get("viewport_camera_height", self.viewport_camera_height)

        print("RL device: ", self.rl_device)

        self._env = env
        self.is_extension = False

        if not hasattr(self, "_num_agents"):
            self._num_agents = 1  # used for multi-agent environments
        if not hasattr(self, "_num_states"):
            self._num_states = 0

        # initialize data spaces (defaults to gym.Box)
        if not hasattr(self, "action_space"):
            self.action_space = spaces.Box(
                np.ones(self.num_actions, dtype=np.float32) * -1.0, np.ones(self.num_actions, dtype=np.float32) * 1.0
            )
        if not hasattr(self, "observation_space"):
            self.observation_space = spaces.Box(
                np.ones(self.num_observations, dtype=np.float32) * -np.Inf,
                np.ones(self.num_observations, dtype=np.float32) * np.Inf,
            )
        if not hasattr(self, "state_space"):
            self.state_space = spaces.Box(
                np.ones(self.num_states, dtype=np.float32) * -np.Inf,
                np.ones(self.num_states, dtype=np.float32) * np.Inf,
            )
        
        self.cleanup()

    def cleanup(self) -> None:
        """Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = torch.zeros((self._num_envs, self.num_observations), device=self._device, dtype=torch.float)
        if self.num_privileged_observations is not None:
            self.privileged_obs_buf = torch.zeros(self._num_envs, self.num_privileged_observations, device=self._device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
        self.obs_history_buf = torch.zeros(self._num_envs, self._obs_history_length, self._num_proprio + self._num_height_points, device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self._num_envs, self.num_states), device=self._device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.extras = {}

    def set_up_scene(
        self, scene, replicate_physics=True, collision_filter_global_paths=[], filter_collisions=True, copy_from_source=False
    ) -> None:
        """Clones environments based on value provided in task config and applies collision filters to mask
            collisions across environments.

        Args:
            scene (Scene): Scene to add objects to.
            replicate_physics (bool): Clone physics using PhysX API for better performance.
            collision_filter_global_paths (list): Prim paths of global objects that should not have collision masked.
            filter_collisions (bool): Mask off collision between environments.
            copy_from_source (bool): Copy from source prim when cloning instead of inheriting.
        """

        super().set_up_scene(scene)

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self.default_zero_env_path)

        if self._task_cfg["sim"].get("add_ground_plane", True):
            self._ground_plane_path = "/World/defaultGroundPlane"
            collision_filter_global_paths.append(self._ground_plane_path)
            scene.add_default_ground_plane(prim_path=self._ground_plane_path)
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=replicate_physics, copy_from_source=copy_from_source
        )
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        if filter_collisions:
            self._cloner.filter_collisions(
                self._env.world.get_physics_context().prim_path,
                "/World/collisions",
                prim_paths,
                collision_filter_global_paths,
            )
        if self._env.render_enabled:
            self.set_initial_camera_params(camera_position=self.camera_position, camera_target=self.camera_target)
            if self._task_cfg["sim"].get("add_distant_light", True):
                self._create_distant_light()
        # initialize capturer for viewport recording
        # this has to be called after initializing replicator for DR
        if self._cfg.get("enable_recording", False):
            self._env.create_viewport_render_product(resolution=(self.viewport_camera_width, self.viewport_camera_height))

    def set_initial_camera_params(self, camera_position, camera_target):
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        viewport_api_2 = get_viewport_from_window_name("Viewport")
        viewport_api_2.set_active_camera("/OmniverseKit_Persp")
        camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport_api_2)
        camera_state.set_position_world(Gf.Vec3d(camera_position[0], camera_position[1], camera_position[2]), True)
        camera_state.set_target_world(Gf.Vec3d(camera_target[0], camera_target[1], camera_target[2]), True)

    def _create_distant_light(self, prim_path="/World/defaultDistantLight", intensity=5000):
        stage = get_current_stage()
        light = UsdLux.DistantLight.Define(stage, prim_path)
        light.CreateIntensityAttr().Set(intensity)

    def initialize_views(self, scene):
        """Optionally implemented by individual task classes to initialize views used in the task.
            This API is required for the extension workflow, where tasks are expected to train on a pre-defined stage.

        Args:
            scene (Scene): Scene to remove existing views and initialize/add new views.
        """
        self._cloner = GridCloner(spacing=self._env_spacing)
        pos, _ = self._cloner.get_clone_transforms(self._num_envs)
        self._env_pos = torch.tensor(np.array(pos), device=self._device, dtype=torch.float)
        if self._env.render_enabled:
            # initialize capturer for viewport recording
            if self._cfg.get("enable_recording", False):
                self._env.create_viewport_render_product(resolution=(self.viewport_camera_width, self.viewport_camera_height))

    @property
    def default_base_env_path(self):
        """Retrieves default path to the parent of all env prims.

        Returns:
            default_base_env_path(str): Defaults to "/World/envs".
        """
        return "/World/envs"

    @property
    def default_zero_env_path(self):
        """Retrieves default path to the first env prim (index 0).

        Returns:
            default_zero_env_path(str): Defaults to "/World/envs/env_0".
        """
        return f"{self.default_base_env_path}/env_0"

    def reset(self):
        """Flags all environments for reset."""
        self.reset_buf = torch.ones_like(self.reset_buf)

    def post_physics_step(self):
        """Processes RL required computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1

        if self._env.world.is_playing():
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    @property
    def world(self):
        """Retrieves the World object for simulation.

        Returns:
            world(World): Simulation World.
        """
        return self._env.world

    @property
    def cfg(self):
        """Retrieves the main config.

        Returns:
            cfg(dict): Main config dictionary.
        """
        return self._cfg

    def set_is_extension(self, is_extension):
        self.is_extension = is_extension

