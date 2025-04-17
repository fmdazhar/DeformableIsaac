# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import datetime
import os
import gym
import hydra
import sys
import torch
from omegaconf import DictConfig
import omniisaacgymenvs
from omniisaacgymenvs.envs.vec_env import VecEnv
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path, get_experience
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task
from rsl_rl.runners import OnPolicyRunner


class RLGTrainer:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self):

        self.cfg_dict["task"]["test"] = self.cfg.test
        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self, env, module_path, experiment_dir):
        log_dir = os.path.join(module_path, "runs")
        os.makedirs(log_dir, exist_ok=True)
        device = self.cfg.rl_device

        if self.cfg.test:
            log_dir = os.path.join(log_dir, "test")

        if self.cfg.wandb_activate:
            # Make sure to install WandB if you actually use this.
            import wandb

            run_name=f"{self.cfg.wandb_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_test" if self.cfg.test else None

            wandb.init(
                project=self.cfg.wandb_project,
                group=self.cfg.wandb_group,
                entity=self.cfg.wandb_entity,
                config=self.cfg_dict,
                sync_tensorboard=True,
                id=run_name,
                resume="allow",
                monitor_gym=True,
                dir=log_dir  # Setting the directory for wandb logs.
            )

        runner = OnPolicyRunner(env, self.rlg_config_dict, log_dir, device=device,
                                wandb_activate=self.cfg.wandb_activate)

        if self.cfg.checkpoint and self.cfg.train.runner.resume:
            print(f"Resuming training from checkpoint: {self.cfg.checkpoint}")
            runner.load(self.cfg.checkpoint)

        if not self.cfg.test:
            runner.learn(num_learning_iterations=self.rlg_config_dict["runner"]["max_iterations"], init_at_random_ep_len=True)

            # dump config dict#
            os.makedirs(experiment_dir, exist_ok=True)
            with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))

        else:
            policy = runner.get_inference_policy(device=device)
            first_frame = True
            step_counter = 0
            max_steps = 10 * int(env.max_episode_length)
            x_vel_cmd = torch.tensor([1.0], device=device)
            y_vel_cmd = torch.tensor([0.0], device=device)
            z_vel_cmd = torch.tensor([0.0], device=device)

            while env.simulation_app.is_running():
                if env.world.is_playing():
                    if first_frame:
                        env.reset()
                        first_frame = False
                    else:
                        if step_counter >= max_steps:
                            break  # stop after a fixed number of steps (optional for test duration)
                        obs = env.get_observations()
                        actions = policy(obs.detach())
                        env._task.commands[:, 0] = x_vel_cmd
                        env._task.commands[:, 1] = y_vel_cmd
                        env._task.commands[:, 2] = z_vel_cmd

                        env.step(actions.detach())
                        if self.cfg.wandb_activate:
                            # Assuming the task (e.g. AnymalTerrainTask) is available as env.task
                            mean_episode_sums = {}
                            for key, tensor in env._task.episode_sums.items():
                                # Compute the mean value across all environments
                                mean_episode_sums[f"test/episode_sum_{key}"] = float(tensor.mean().item())
                            # Optionally, log additional test metrics here.
                            wandb.log(mean_episode_sums, step=step_counter)
                        step_counter += 1
                else:
                    env.world.step(render=not self.cfg.headless)
        env.simulation_app.close()

@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    headless = cfg.headless

    # process additional kit arguments and write them to argv
    if cfg.extras and len(cfg.extras) > 0:
        sys.argv += cfg.extras

    # local rank (GPU id) in a current multi-gpu mode
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # global rank (GPU id) in multi-gpu multi-node mode
    global_rank = int(os.getenv("RANK", "0"))

    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # select kit app file
    experience = get_experience(headless, cfg.enable_livestream, enable_viewport, cfg.enable_recording, cfg.kit_app)

    env = VecEnv(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport or cfg.enable_recording,
        experience=experience
    )

    # parse experiment directory
    module_path = os.path.abspath(os.path.join(os.path.dirname(omniisaacgymenvs.__file__)))
    experiment_dir = os.path.join(module_path, "runs", cfg.train.params.config.name)

    # use gym RecordVideo wrapper for viewport recording
    if cfg.enable_recording:
        if cfg.recording_dir == '':
            videos_dir = os.path.join(experiment_dir, "videos")
        else:
            videos_dir = cfg.recording_dir
        video_interval = lambda step: step % cfg.recording_interval == 0
        video_length = cfg.recording_length
        env.is_vector_env = True
        if env.metadata is None:
            env.metadata = {"render_modes": ["rgb_array"], "render_fps": cfg.recording_fps}
        else:
            env.metadata["render_modes"] = ["rgb_array"]
            env.metadata["render_fps"] = cfg.recording_fps
        env = gym.wrappers.RecordVideo(
            env, video_folder=videos_dir, step_trigger=video_interval, video_length=video_length
        )

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    # Optionally override config values in test mode:
    if cfg.test:
        # Overwrite specific config entries for test scenarios:
        cfg.task.env.numEnvs = 4
        cfg.task.env.terrain.numLevels = 1
        cfg.task.env.terrain.numTerrains = 1
        cfg.task.env.terrain.curriculum = False
        cfg.task.env.terrain.VelocityCurriculum = False
        cfg.task.env.terrain.terrain_types = 1

        cfg.task.env.terrain.terrain_types = [
            {
                "name": "flat",
                "particle_present": False,
                "compliant": False,
                "count": 2,
                "level": 0
            }
        ]
        cfg.task.env.learn.addNoise = False
        cfg.task.env.learn.randomizationRanges.randomizeGravity = False
        cfg.task.env.learn.randomizationRanges.randomizeFriction = False
        cfg.task.env.learn.randomizationRanges.randomizeCOM = False
        cfg.task.env.learn.randomizationRanges.randomizeAddedMass = False
        cfg.task.env.learn.randomizationRanges.randomizeMotorStrength = False
        cfg.task.env.learn.randomizationRanges.randomizeMotorOffset = False
        cfg.task.env.learn.randomizationRanges.randomizeKpFactor = False
        cfg.task.env.learn.randomizationRanges.randomizeKdFactor = False
        cfg.task.env.learn.randomizationRanges.material_randomization.enabled = False  
        # Add any other overrides you need.


    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = cfg.seed + global_rank if cfg.seed != -1 else cfg.seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed

    task = initialize_task(cfg_dict, env)

    torch.cuda.set_device(local_rank)
    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra()
    rlg_trainer.run(env, module_path, experiment_dir)
    env.close()



    if cfg.wandb_activate and global_rank == 0:
        wandb.finish()
    


if __name__ == "__main__":
    parse_hydra_configs()
