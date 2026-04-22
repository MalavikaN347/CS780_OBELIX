"""
Submission agent (USES trained weights).

Trained with Double DQN via train.py — place weights.pth in the same folder.
The evaluator imports this file and calls `policy(obs, rng)`.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # cached after first load


def _load_once():
    """Load the trained Double DQN network and weights (runs only once)."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    import torch
    import torch.nn as nn

    class Net(nn.Module):
        """
        Must match the architecture in train.py exactly so that
        torch.load_state_dict() works without any key mismatches.
        """
        def __init__(self, obs_dim: int = 18, n_actions: int = 5, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x):
            return self.net(x)

    model = Net()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Choose the best action from the 18-dim OBELIX observation.

    Parameters
    ----------
    obs : np.ndarray, shape (18,)
        Sensor feedback vector returned by the environment.
    rng : np.random.Generator
        Provided by the evaluator — not used (policy is deterministic).

    Returns
    -------
    str
        One of: "L45", "L22", "FW", "R22", "R45"
    """
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)  # (1, 18)
    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0).numpy()                # (5,)

    return ACTIONS[int(np.argmax(q_values))]