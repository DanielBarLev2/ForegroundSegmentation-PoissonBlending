import numpy as np

# define the eight directions for neighbors (bidirectional way)
EIGHT_DIR = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)])

HARD_BG = 0  # Hard bg pixel
HARD_FG = 1  # Hard fg pixel, will not be used
SOFT_BG = 2  # Soft bg pixel
SOFT_FG = 3  # Soft fg pixel

# epsilon for numerical stability
EPSILON = 1e-4
