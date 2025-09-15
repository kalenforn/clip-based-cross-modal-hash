import random
import numpy as np
import torch


def set_seed(seed=1814, logger=None):

    if logger is not None:
        logger.info(f"Setting seed: {seed}.")
    else:
        print(f"Setting seed: {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    