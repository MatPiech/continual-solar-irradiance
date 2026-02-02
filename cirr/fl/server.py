import copy

import torch


def federated_averaging(global_weights, client_updates):
    """Perform federated averaging on client model updates.

    Parameters
    ----------
    global_weights : dict
        The old model state dictionary.
    client_updates : list
        List of state_dicts from clients.

    Returns
    -------
    new_weights : dict
        Averaged model weights.
    weight_change_mb : float
        L2 norm of global weight change.
    """
    if not client_updates:
        return global_weights, 0.0

    # Simple averaging
    new_weights = copy.deepcopy(global_weights)

    for key in new_weights:
        # Stack the tensors from all updates and take mean
        tensors = [update[key] for update in client_updates]

        # Only average floating-point tensors; keep first value for others (e.g., counters)
        if tensors[0].is_floating_point():
            new_weights[key] = torch.stack(tensors).mean(dim=0)
        else:
            new_weights[key] = tensors[0]

    # Compute weight change (global model change)
    total_change = 0.0
    for key in global_weights:
        if global_weights[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            diff = global_weights[key].float() - new_weights[key].float()
            total_change += torch.sum(diff**2).item()
    weight_change_mb = total_change**0.5

    return new_weights, weight_change_mb
