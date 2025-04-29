


import os

import carb
from hydra.utils import to_absolute_path


def is_valid_local_file(path):
    return os.path.isfile(path)


def is_valid_ov_file(path):
    import omni.client

    result, entry = omni.client.stat(path)
    return result == omni.client.Result.OK


def download_ov_file(source_path, target_path):
    import omni.client

    result = omni.client.copy(source_path, target_path)

    if result == omni.client.Result.OK:
        return True
    return False


def break_ov_path(path):
    import omni.client

    return omni.client.break_url(path)


def retrieve_checkpoint_path(path):
    # check if it's a local path
    if is_valid_local_file(path):
        return to_absolute_path(path)
    # check if it's an OV path
    elif is_valid_ov_file(path):
        ov_path = break_ov_path(path)
        file_name = os.path.basename(ov_path.path)
        target_path = f"checkpoints/{file_name}"
        copy_to_local = download_ov_file(path, target_path)
        return to_absolute_path(target_path)
    else:
        carb.log_error(f"Invalid checkpoint path: {path}. Does the file exist?")
        return None


def get_experience(headless, enable_livestream, enable_viewport, enable_recording, kit_app):
    if kit_app == '':
        if enable_viewport:
            import leggeedisaacgymenvs
            experience = os.path.abspath(os.path.join(os.path.dirname(leggeedisaacgymenvs.__file__), '../apps/omni.isaac.sim.python.gym.camera.kit'))
        else:
            experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit'
            if enable_livestream:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.livestream.kit'
            elif headless and not enable_recording:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'
    else:
        experience = kit_app
    return experience
