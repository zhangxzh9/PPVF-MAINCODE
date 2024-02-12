import numpy as np
import torch
import random
from utils.env import Env
from utils.config import Config
from utils.trace import Trace

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    cfg   = Config()

    trace = Trace(cfg=cfg)
    env   = Env(trace = trace, cfg=cfg)

    env.main_Test()