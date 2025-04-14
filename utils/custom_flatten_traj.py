from typing import Iterable, Mapping, List, Any
import numpy as np
from imitation.data import types


def flatten_trajectories_with_transpose(
    trajectories: Iterable[types.Trajectory],
) -> types.Transitions:
    """Flatten trajectories into transitions and transpose obs (e.g. from HWC to CHW).

    This version transposes observations from (H, W, C) to (C, H, W) format.

    Args:
        trajectories: list of trajectory objects to flatten.

    Returns:
        A single Transitions object with all trajectories flattened and observations transposed.
    """
    def all_of_type(key, desired_type):
        return all(isinstance(getattr(traj, key), desired_type) for traj in trajectories)

    assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)

    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}

    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs


        ################## [Revised] Transpose the observation shape ########################

        if obs.ndim == 4 and obs.shape[-1] == 3:   # e.g., (T+1, 96, 96, 3)
            obs = np.transpose(obs, (0, 3, 1, 2))  # (T+1, C, H, W)
        
        #####################################################################################

        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        if traj.infos is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = traj.infos
        parts["infos"].append(infos)

    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }

    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"

    return types.Transitions(**cat_parts)
