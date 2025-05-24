


from datetime import datetime

import numpy as np
import torch
from .vec_env_base import VecEnvBase


# VecEnv Wrapper for RL training
class VecEnv(VecEnvBase):
    def _process_data(self):
        self._obs = self._obs.to(self._task.rl_device)
        if self._priv_obs is not None:
            self._priv_obs = self._priv_obs.to(self._task.rl_device)
        self._rew = self._rew.to(self._task.rl_device)
        self._resets = self._resets.to(self._task.rl_device)
        self._extras = self._extras

    def set_task(self, task, backend="numpy", sim_params=None, init_sim=True, rendering_dt=1.0 / 60.0) -> None:
        super().set_task(task, backend, sim_params, init_sim, rendering_dt)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space
        self.num_obs = self._task._num_observations
        self.num_height_points = self._task._num_height_points
        self.num_privileged_obs = self._task._num_privileged_observations
        self.num_proprio = self._task._num_proprio
        self.num_scan = self._task._num_height_points
        self.num_priv = self._task._num_priv
        self.history_len = self._task._obs_history_length
        self.num_obs_history = self._task._num_obs_history
        self.num_actions = self._task._num_actions
        self.max_episode_length = self._task.max_episode_length
        self.episode_length_buf = self._task.progress_buf
        self.dt = self._task.dt

        print(f"VecEnv: num_states={self.num_states}, num_observations={self.num_obs}, "
              f"num_privileged_observations={self.num_privileged_obs}, num_obs_history={self.num_obs_history}, "
              f"num_actions={self.num_actions}, max_episode_length={self.max_episode_length}", 
              f"num_height_points={self.num_height_points}, num_proprio={self.num_proprio}, "
              f"num_priv={self.num_priv}, history_len={self.history_len}")

    def step(self, actions):
        # only enable rendering when we are recording, or if the task already has it enabled
        to_render = self._render
        if self._record:
            if not hasattr(self, "step_count"):
                self.step_count = 0
            if self.step_count % self._task.cfg["recording_interval"] == 0:
                self.is_recording = True
                self.record_length = 0
            if self.is_recording:
                self.record_length += 1
                if self.record_length > self._task.cfg["recording_length"]:
                    self.is_recording = False
            if self.is_recording:
                to_render = True
            else:
                if (self._task.cfg["headless"] and not self._task.enable_cameras and not self._task.cfg["enable_livestream"]):
                    to_render = False
            self.step_count += 1

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device)

        self._task.pre_physics_step(actions)

        if (self.sim_frame_count + self._task.control_frequency_inv) % self._task.rendering_interval == 0:
            for _ in range(self._task.control_frequency_inv - 1):
                self._world.step(render=False)
                self.sim_frame_count += 1
            self._world.step(render=to_render)
            self.sim_frame_count += 1
        else:
            for _ in range(self._task.control_frequency_inv):
                self._world.step(render=False)
                self.sim_frame_count += 1

        self._obs, self._priv_obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        self._process_data()
        return self._obs, self._priv_obs, self._rew, self._resets, self._extras

    def get_observations(self):
        return self._task.obs_buf
    
    def get_privileged_observations(self):
        return self._task.privileged_obs_buf

    def reset(self, seed=None, options=None):
        """Resets the task and applies default zero actions to recompute observations and states."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.rl_device)
        obs,_, _, _, _ = self.step(actions)

        return obs
