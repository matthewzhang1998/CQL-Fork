import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import json
import robomimic.utils.obs_utils
from robomimic.config import config_factory

import os
import numpy as np


PERMITTED_KEYS = [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",]  # duplicated from data_utils, should clean up


class DummySpace:
    def __init__(self, shape):
        self.shape = shape


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs_dict, r, dones, infos = self.env.step(action)
        obs = process_obs_dict(obs_dict)

        return obs, r, dones, infos

    def reset(self):
        obs_dict = self.env.reset()
        obs = process_obs_dict(obs_dict)

        return obs

def process_obs_dict(obs_dict):
    temp = []
    for key in PERMITTED_KEYS:  # ensure consistent ordering
        assert key in obs_dict
        temp.append(obs_dict[key])

    temp = np.hstack(temp)
    return temp


def make(data_path, config_path):
    print(os.getcwd())
    print(os.path.exists(config_path))
    ext_cfg = json.load(open(config_path, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)

    config.lock()

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=data_path)


    ObsUtils.initialize_obs_utils_with_config(config)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=data_path,
        all_modalities=config.all_modalities,
        verbose=True
    )

    env = EnvUtils.create_env_from_metadata(env_meta, "Lift", render=False, render_offscreen=False)

    env = EnvWrapper(env)

    env.observation_space = DummySpace([19])  # static for now
    env.action_space = DummySpace([7])

    return env
