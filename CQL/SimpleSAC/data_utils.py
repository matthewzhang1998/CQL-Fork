import argparse
import h5py
from collections import defaultdict

import numpy as np
import torch.utils.data.dataset

import sklearn.model_selection

PERMITTED_KEYS = [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",] # just these ones I guess

REMAP_ENUM = {"obs": "observations", "next_obs": "next_observations"}

class GANDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class IMDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def load_data(data_path):
    # code from Robomimic
    with h5py.File(data_path, "r") as f:
        keys = list(f.keys())[0]

        # Get the data
        data_dict = dict(f[keys])

        return_dict = defaultdict(lambda: [])
        for demo in data_dict:
            demo_dict = {}
            # static get
            # Question: keep states? (oracle)
            for x in ['actions', 'dones', 'rewards', 'states']:
                if x in data_dict[demo]:
                    demo_dict[x] = data_dict[demo][x][:]
            for y in ['obs', 'next_obs']:
                temp = []
                for obs_z in PERMITTED_KEYS:
                    assert obs_z in data_dict[demo][y].keys()
                    temp.append(data_dict[demo][y][obs_z])

                temp = np.hstack(temp)
                demo_dict[REMAP_ENUM[y]] = temp

            for key in demo_dict:
                return_dict[key].append(demo_dict[key])

    return return_dict


def whiten_data(data):
    #mu = np.mean(data, axis=0, keepdims=True)
    #std = np.std(data, axis=0, keepdims=True)
    mu = 0
    std = 1

    white_data = (data - mu)/std

    white_stats = {"mean": mu, "standard_deviation": std}

    return white_data, white_stats


def prep_im_data(raw_data):
    # for imitation learning, actions and states only
    white_stats = {}
    raw_data['obs'] = [i[0] for i in raw_data['obs']]
    raw_data['actions'] = [i[0] for i in raw_data['actions']]

    obs = np.concatenate(raw_data['obs'], axis=0)
    actions = np.concatenate(raw_data['actions'], axis=0)

    white_obs, white_obs_stats = whiten_data(obs)
    white_stats["obs"] = white_obs_stats
    white_actions, white_action_stats = whiten_data(actions)
    white_stats['actions'] = white_action_stats
    # compress white stats??
    metadata = get_metadata(obs, actions)

    #return np.hstack([white_obs, white_actions]), white_stats, metadata
    return list(zip(white_obs, white_actions)), white_stats, metadata


def make_cql_dataset(data_path):
    return load_data(data_path)


def make_im_dataset(data_path, batch_size, train_test_split):
    raw_data = load_data(data_path)
    data_array, white_stats, metadata = prep_im_data(raw_data)

    if train_test_split:
        train_data, test_data = sklearn.model_selection.train_test_split(data_array, test_size=train_test_split)
        train_data = GANDataset(train_data)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        test_data = GANDataset(test_data)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
        return (train_loader, test_loader), white_stats, metadata
    else:
        train_data = GANDataset(data_array)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader, white_stats, metadata


def get_metadata(obs, actions):
    obs_dim = np.shape(obs)[-1]
    action_dim = np.shape(actions)[-1]

    return {"obs_dim": obs_dim, "action_dim": action_dim}


def make_brl_data(raw_data):
    # TODO: more complicated here for batch RL
    # construct pseudo-reward for human trajectories?
    # keep rewards, dones, next states?
    # possibly not enough data for GAN to really work (high dim)
    pass


class Sampler:
    def __init__(self, dataset):
        self.data = dataset

    def sample(self, batch_size):
        # just sample randomly for now
        next_obs, next_actions = next(iter(self.data))
        return next_obs.detach().numpy(), next_actions.detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/mg/low_dim_sparse.hdf5")

    args = parser.parse_args()
    argdict = vars(args)

    print(argdict)
    data = load_data(argdict["data_path"])
    for i in data["obs"]:
        if (i[0].shape[-1]) != 55:
            print(i[0].shape[-1])