from copy import deepcopy
import numpy as np
from utils.utils import to_torch_type, to_numpy_type


def add_group_info(groups, scheme):
    for name, info in scheme.items():
        vshape = info["vshape"]
        group = info.get("group", None)
        if isinstance(vshape, int):
            vshape = (vshape,)
        if group:
            vshape = (groups[group], *vshape)
        info["vshape"] = vshape
    return scheme


def make_buffer_scheme(args):
    buffer_scheme = deepcopy(args.scheme)
    groups = args.groups
    # add group info
    buffer_scheme = add_group_info(groups, buffer_scheme)
    # add extra data
    buffer_scheme.update({"filled": {"vshape": (1,), "dtype": np.int32},
                          "actions_onehot": {"vshape": (args.n_agents, args.n_actions,), "dtype": np.int64}})
    if "rnn" in args.agent:
        buffer_scheme.update({"hidden_states": {"vshape": (args.n_agents, args.hidden_dim,), "dtype": np.float32}})
    if args.save_probs:
        buffer_scheme.update({"probs": {"vshape": (args.n_agents, args.n_actions,), "dtype": np.float32}})
    # add length
    for _, info in buffer_scheme.items():
        info["vshape"] = (args.episode_length, *info["vshape"])
        # to torch type
        info["dtype"] = to_torch_type(info["dtype"])

    return buffer_scheme


# zmq scheme
def make_episode_scheme(args):
    episode_scheme = make_buffer_scheme(args)
    episode_scheme.pop("hidden_state", None)
    for _, info in episode_scheme.items():
        info["dtype"] = to_numpy_type(info["dtype"])
    return episode_scheme


# zmq scheme
# todo change batch to single
# todo change actions shape
def make_transition_scheme(args):
    transition_scheme = deepcopy(args.scheme)
    groups = args.groups
    # add group info
    transition_scheme = add_group_info(groups, transition_scheme)

    for _, info in transition_scheme.items():
        info["vshape"] = (args.env_batch_size, *info["vshape"])

    for k in transition_scheme.keys():
        if k == "actions":
            transition_scheme[k]['zmq_send'] = 'learner'
        else:
            transition_scheme[k]['zmq_send'] = 'actor'
    return transition_scheme


scheme_REGISTRY = {}
scheme_REGISTRY["buffer"] = make_buffer_scheme
scheme_REGISTRY["episode"] = make_episode_scheme
scheme_REGISTRY["transition"] = make_transition_scheme
