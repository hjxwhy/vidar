# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os
import sys
sys.path.append(os.getcwd())
import fire
import torch

from vidar.core.trainer import Trainer
from vidar.core.wrapper import Wrapper
from vidar.utils.config import read_config


def train(cfg='configs/papers/fsm/train_ddad.yaml', **kwargs):

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    cfg = read_config(cfg, **kwargs)

    wrapper = Wrapper(cfg, verbose=True)
    trainer = Trainer(cfg)
    trainer.learn(wrapper)


if __name__ == '__main__':
    fire.Fire(train)
