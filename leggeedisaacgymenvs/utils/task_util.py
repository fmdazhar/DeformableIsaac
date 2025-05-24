


def import_tasks():

    from leggeedisaacgymenvs.tasks.anymal_terrain import AnymalTerrainTask

    # Mappings from strings to environments
    task_map = {
        "AnymalTerrain": AnymalTerrainTask,
    }


    return task_map


def initialize_task(config, env, init_sim=True):
    from leggeedisaacgymenvs.utils.config_utils.sim_config import SimConfig

    sim_config = SimConfig(config)
    task_map = import_tasks()

    cfg = sim_config.config

    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    backend = "torch"

    rendering_dt = sim_config.get_physics_params()["rendering_dt"]

    env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend=backend,
        init_sim=init_sim,
        rendering_dt=rendering_dt,
    )

    return task
