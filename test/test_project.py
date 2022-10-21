import sys
import os
sys.path.append(os.getcwd())
sys.path.append('/home/hjx/vidar/display')
from display.display_sample import display_sample
from vidar.utils.config import read_config
from vidar.utils.data import set_random_seed
from vidar.utils.setup import setup_datasets

set_random_seed(42)

cfg = read_config('/home/hjx/vidar/configs/papers/fsm/train_ddad.yaml')
datasets, _ = setup_datasets(cfg.datasets, verbose=True)
datasets = datasets['train']
