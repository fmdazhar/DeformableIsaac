import datetime
import os
import gym
import hydra
import sys
import torch
import numpy as np
from omegaconf import DictConfig
import leggeedisaacgymenvs
from leggeedisaacgymenvs.envs.vec_env import VecEnv
from leggeedisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path, get_experience
from leggeedisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from leggeedisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from leggeedisaacgymenvs.utils.task_util import initialize_task
from rsl_rl.runners import OnPolicyRunner


class Trainer:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self):

        self.cfg_dict["task"]["test"] = self.cfg.test
        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self, env, experiment_dir):
        log_dir = experiment_dir
        os.makedirs(log_dir, exist_ok=True)
        device = self.cfg.rl_device

        if self.cfg.test:
            log_dir = os.path.join(log_dir, "test")
            os.makedirs(log_dir, exist_ok=True)

        if self.cfg.wandb_activate:
            # Make sure to install WandB if you actually use this.
            import wandb

            run_name=f"{self.cfg.wandb_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_test" if self.cfg.test else None

            wandb.init(
                project=self.cfg.wandb_project,
                group=self.cfg.wandb_group,
                entity=self.cfg.wandb_entity,
                config=self.cfg_dict,
                id=run_name,
                resume="allow",
                monitor_gym=True,
                dir=log_dir  # Setting the directory for wandb logs.
            )


        runner = OnPolicyRunner(env, self.rlg_config_dict, log_dir, device=device,
                                wandb_activate=self.cfg.wandb_activate)

        if self.cfg.checkpoint :
            print(f"Resuming training from checkpoint: {self.cfg.checkpoint}")
            runner.load(self.cfg.checkpoint)

        if not self.cfg.test:
            runner.learn(num_learning_iterations=self.rlg_config_dict["runner"]["max_iterations"])

            # dump config dict#
            os.makedirs(experiment_dir, exist_ok=True)
            with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))

        else:
            policy = runner.get_inference_policy(device=device)
            step_counter = 0
            num_episodes = 10
            max_steps = num_episodes * int(env.max_episode_length)
            print(f"Running test for {max_steps} steps.")
            x_vel_cmd = torch.tensor([1.0], device=device)
            y_vel_cmd = torch.tensor([0.0], device=device)
            z_vel_cmd = torch.tensor([0.0], device=device)

            while env.simulation_app.is_running():
                if env.world.is_playing():
                    if step_counter >= max_steps:
                        print("Test duration reached. Stopping the simulation.")
                        break  # stop after a fixed number of steps (optional for test duration)
                    obs = env.get_observations().to(device)
                    actions = policy(obs.detach())
                    env._task.commands[:, 0] = x_vel_cmd
                    env._task.commands[:, 1] = y_vel_cmd
                    env._task.commands[:, 2] = z_vel_cmd

                    env.step(actions.detach())
                    if self.cfg.wandb_activate:
                        log_data = {
                            f"test/episode_sum_{k}": v.mean().item()
                            for k, v in env._task.episode_sums_raw.items()
                        }

                        lin_err = (env._task.base_lin_vel[:, 0] - env._task.commands[:, 0]).cpu()
                        ang_err = (env._task.base_ang_vel[:, 2] - env._task.commands[:, 2]).cpu()
                        log_data["test/lin_vel_x_rmsd"]  = torch.sqrt((lin_err ** 2).mean()).item()
                        log_data["test/ang_vel_rmsd"]  = torch.sqrt((ang_err ** 2).mean()).item()
                        
                        power_consumption = torch.sum(torch.multiply(env._task.torques, env._task.dof_vel), dim=1).cpu()
                        log_data["test/mean_power_consumption"] = power_consumption.mean().item()
                        
                        max_torque, max_torque_indices = torch.max(torch.abs(env._task.torques), dim=1)
                        log_data["test/mean_max_torque"] = max_torque.cpu()

                        m = (env._task.total_masses + env._task.payloads).cpu()
                        g = 9.8  # m/s^2
                        v = torch.norm(env._task.base_lin_vel[:, 0:2], dim=1).cpu()
                        cot = power_consumption / (m * g * v.clamp(min=1e-3))
                        log_data["test/mean_cot"] = cot.mean().item()

                        h = 0.28 #m
                        froude_number =  env._task.base_lin_vel[:, 0].cpu() ** 2 / (g * h)
                        log_data["test/mean_froude_number"] = froude_number.mean().item()

                        # Log the foot contact forces
                        foot_forces = env._task.foot_contact_forces.detach().cpu()    # Nx4x3
                        foot0 = foot_forces[0]
                        foot0_norms = torch.norm(foot0, dim=1)       # Nx4
                        norm = torch.sum((torch.norm(foot_forces[:, env._task.feet_indices.cpu(), :], dim=-1)).clip(min=0.), dim=1)
                        log_data["test/foot0_norm"] = norm[0].item()
                        feet = ["FL", "FR", "RL", "RR"]
                        for i, foot_name in enumerate(feet):
                            # log each component
                            fx, fy, fz = foot0[i].tolist()
                            log_data[f"test/foot0_{foot_name}_fx"] = fx
                            log_data[f"test/foot0_{foot_name}_fy"] = fy
                            log_data[f"test/foot0_{foot_name}_fz"] = fz
                            log_data[f"test/foot0_{foot_name}_norm"] = foot0_norms[i].item()

                        wandb.log(log_data, step=step_counter)
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
    module_path = os.path.abspath(os.path.join(os.path.dirname(leggeedisaacgymenvs.__file__)))
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
        cfg.task.env.numEnvs = 1
        cfg.task.env.terrain.numLevels = 10
        cfg.task.env.terrain.numTerrains = 10
        cfg.task.env.terrain.measureHeights = True


        cfg.task.env.terrain.curriculum = False
        cfg.task.env.terrain.flat = False
        cfg.task.env.terrain.debugHeights = True
        cfg.task.env.commands.VelocityCurriculum = True
        cfg.task.env.terrain.oobActive = True

        cfg.task.env.terrain.terrain_types = [
            # {
            #     "name": "flat",
            #     "particle_present": False,
            #     "compliant": False,
            #     "row_count": 2,
            #     "col_count": 2,
            #     "level": 0,
            # },
            # {
            #     "name": "flat",
            #     "particle_present": False,
            #     "compliant": True,
            #     "row_count": 2,
            #     "col_count": 2,
            #     "level": 0,
            # },
            {
                "name": "central_depression_terrain",
                "compliant": True,
                "level": 0,
                "particle_present": True,
                "system": 1,
                "depth": 0.10,
                "size": 8,
                "row_count": 1,
                "col_count": 1,
            },
            # {
            #     "name": "central_depression_terrain",
            #     "compliant": True,
            #     "level": 0,
            #     "particle_present": True,
            #     "system": 3,
            #     "depth": 0.15,
            #     "size": 4,
            #     "row_count": 1,
            #     "col_count": 1,
            # }
        ]

        cfg.task.env.terrain.particles.resetIntervalSecs = [4,10]

        cfg.task.env.learn.addNoise = False
        cfg.task.env.learn.episodeLength_s = 20
        cfg.task.env.randomizationRanges.randomizeEpisodeStart = False

        cfg.task.env.randomizationRanges.randomizeGravity = False
        cfg.task.env.randomizationRanges.randomizeFriction = False
        cfg.task.env.randomizationRanges.randomizeCOM = False
        cfg.task.env.randomizationRanges.randomizeAddedMass = False
        cfg.task.env.randomizationRanges.randomizeMotorStrength = False
        cfg.task.env.randomizationRanges.randomizeMotorOffset = False
        cfg.task.env.randomizationRanges.randomizeKpFactor = False
        cfg.task.env.randomizationRanges.randomizeKdFactor = False

        # cfg.task.env.randomizationRanges.material_randomization.particles.enabled = True 
        # cfg.task.env.randomizationRanges.material_randomization.particles.interval = 1.0 
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
    rlg_trainer = Trainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra()
    rlg_trainer.run(env, experiment_dir)
    env.close()

    if cfg.wandb_activate and global_rank == 0:
        wandb.finish()
    


if __name__ == "__main__":
    parse_hydra_configs()
