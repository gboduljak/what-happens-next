
from collections import namedtuple

Batch = namedtuple(
    "Batch",
    ["frames", "trajectories", "trajectory_masks", "visibility", "index"]
)
