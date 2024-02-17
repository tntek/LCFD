# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information
# _C.SETTING = "continual"
# _C.SETTING = "reset_each_shift"
# _C.SETTING = "lcfd"
# Data directory
_C.DATA_DIR = "/media/ts/tntbak2/Datazoom"

# Weight directory
_C.CKPT_DIR = "/media/ts/tntbak2/Modelzoom"

# GPU id
_C.GPU_ID = '0'
# Output directory
_C.SAVE_DIR = "./output"

_C.ISSAVE = False
# Path to a specific checkpoint
_C.CKPT_PATH = ""

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Optional description of a config
_C.DESC = ""

_C.DA = "uda"

_C.FOLDER = './data/'

_C.NUM_WORKERS = 4

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Some of the available models can be found here:
# Torchvision: https://pytorch.org/vision/0.14/models.html
# timm: https://github.com/huggingface/pytorch-image-models/tree/v0.6.13
# RobustBench: https://github.com/RobustBench/robustbench
_C.MODEL.ARCH = 'Standard'

# Type of pre-trained weights for torchvision models. See: https://pytorch.org/vision/0.14/models.html
_C.MODEL.METHOD = "lcfd"

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

# ----------------------------- SETTING options -------------------------- #
_C.SETTING = CfgNode()

# Dataset for evaluation
_C.SETTING.DATASET = 'office-home'

# The index of source domain
_C.SETTING.S = 0 
# The index of Target domain
_C.SETTING.T = 1

#Seed
_C.SETTING.SEED = 2021

#Sorce model directory
_C.SETTING.OUTPUT_SRC = 'weight_512/seed2021'

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Choices: Adam, SGD
_C.OPTIM.METHOD = "SGD"

# Learning rate
_C.OPTIM.LR = 1e-3

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 5e-4

_C.OPTIM.LR_DECAY1 = 0.1

_C.OPTIM.LR_DECAY2 = 1

_C.OPTIM.LR_DECAY3 = 0.01

# ------------------------------- Test options ------------------------- #
_C.TEST = CfgNode()


# Batch size
_C.TEST.BATCH_SIZE = 64

# Max epoch 
_C.TEST.MAX_EPOCH = 15

# Interval
_C.TEST.INTERVAL = 15

# --------------------------------- SOURCE options ---------------------------- #
_C.SOURCE = CfgNode()

_C.SOURCE.EPSILON = 1e-5

_C.SOURCE.TRTE = 'val'
# --------------------------------- NRC options --------------------------- #
_C.NRC = CfgNode()

_C.NRC.K = 5

_C.NRC.KK = 4

_C.NRC.EPSILON = 1e-5

# --------------------------------- SHOT options ---------------------------- #
_C.SHOT = CfgNode()

_C.SHOT.CLS_PAR = 0.3
_C.SHOT.ENT = True
_C.SHOT.GENT = True
_C.SHOT.EPSILON = 1e-5
_C.SHOT.ENT_PAR = 1.0
_C.SHOT.THRESHOLD = 0.0
_C.SHOT.DISTANCE = 'cosine'# ["cosine", "euclidean"]

# --------------------------------- GKD options ---------------------------- #
_C.GKD = CfgNode()

_C.GKD.CLS_PAR = 0.3
_C.GKD.ENT = True
_C.GKD.GENT = True
_C.GKD.EPSILON = 1e-5
_C.GKD.ENT_PAR = 1.0
_C.GKD.THRESHOLD = 0.0
_C.GKD.DISTANCE = 'cosine'# ["cosine", "euclidean"]

# --------------------------------- TPDS options ---------------------------- #
_C.TPDS = CfgNode()

_C.TPDS.EPSILON = 1e-5
_C.TPDS.THRESHOLD = 0.0
_C.TPDS.DISTANCE = 'cosine'# ["cosine", "euclidean"]

# --------------------------------- COWA options ----------------------------- #
_C.COWA = CfgNode()

_C.COWA.ALPHA = 0.2
_C.COWA.WARM = 0.0
_C.COWA.COEFF = 'JMDS' #['LPG', 'JMDS', 'PPL','NO']
_C.COWA.EPSILON = 1e-5
_C.COWA.EPSILON2 = 1e-6
_C.COWA.DISTANCE = 'cosine'# ["cosine", "euclidean"]
_C.COWA.PICKLE = False

# --------------------------------- PLUE options --------------------- #
_C.PLUE = CfgNode()

_C.PLUE.QUEUE_SIZE = 16384
_C.PLUE.CONTRAST_TYPE = "class_aware"
_C.PLUE.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.PLUE.ALPHA = 1.0  # lambda for classification loss
_C.PLUE.BETA = 1.0   # lambda for instance loss
_C.PLUE.ETA = 1.0    # lambda for diversity loss

_C.PLUE.DIST_TYPE = "cosine" # ["cosine", "euclidean"]
_C.PLUE.CE_SUP_TYPE = "weak_strong" # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.PLUE.REFINE_METHOD = "nearest_neighbors"
_C.PLUE.NUM_NEIGHBORS = 10


# ---------------------------------  options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.QUEUE_SIZE = 16384
_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0  # lambda for classification loss
_C.ADACONTRAST.BETA = 1.0   # lambda for instance loss
_C.ADACONTRAST.ETA = 1.0    # lambda for diversity loss

_C.ADACONTRAST.DIST_TYPE = "cosine" # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong" # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- lcfd options ----------------------------- #
_C.lcfd = CfgNode()

_C.lcfd.CLS_PAR = 0.4
_C.lcfd.LOSS_FUNC = 'sce' #['l1',''l2','kl','sce']
_C.lcfd.ENT = True
_C.lcfd.GENT = True
_C.lcfd.EPSILON = 1e-5
_C.lcfd.GENT_PAR = 1.0
_C.lcfd.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.lcfd.N_CTX = 4 
_C.lcfd.NAME_FILE = 'SHOT/data/office-home/classname.txt' #classname file location
_C.lcfd.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.lcfd.TTA_STEPS = 1
# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args():
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Evaluate")
    # parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        # help="Config file location")
    parser.add_argument("--cfg", dest="cfg_file",default="cfgs/office-home/lcfd.yaml", type=str,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    args = parser.parse_args()
    # cfg = args.cfg
    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    if cfg.SETTING.DATASET == 'office-home':
        cfg.domain = ['Art', 'Clipart', 'Product', 'RealWorld']
        cfg.class_num = 65 
        cfg.name_file = './data/office-home/classname.txt'
        cfg.bottleneck = 512
    if cfg.SETTING.DATASET == 'VISDA-C':
        cfg.domain = ['train', 'validation']
        cfg.class_num = 12
        cfg.name_file = './data/VISDA-C/classname.txt'
        cfg.bottleneck = 512

    cfg.output_dir_src = os.path.join(cfg.CKPT_DIR,cfg.SETTING.OUTPUT_SRC,cfg.DA,cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper())
    cfg.output_dir = os.path.join(cfg.SAVE_DIR,cfg.DA,cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper(),cfg.MODEL.METHOD)
    cfg.name = cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper()
    cfg.name_src = cfg.domain[cfg.SETTING.S][0].upper()
    # cfg.output_dir = os.path.join(cfg.SAVE_DIR,cfg.DA,cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.T][0].upper())
    g_pathmgr.mkdirs(cfg.output_dir)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    # cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.output_dir, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    # if cfg.SETTING.SEED:


    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda
               ]
    logger.info("PyTorch Version: torch={}, cuda={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "DomainNet-126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               "imagenet_v": "imagenetv2-matched-frequency-format-val"
               }
    return os.path.join(root, mapping[dataset_name])


def get_num_classes(dataset_name):
    dataset_name2num_classes = {"cifar10": 10, "cifar10_c": 10, "cifar100": 100,  "cifar100_c": 100,
                                "imagenet": 1000, "imagenet_c": 1000, "imagenet_k": 1000, "imagenet_r": 200,
                                "imagenet_a": 200, "imagenet_d": 164, "imagenet_d109": 109, "imagenet200": 200,
                                "domainnet126": 126, "office31": 31, "visda": 12
                                }
    return dataset_name2num_classes[dataset_name]


def get_domain_sequence(ckpt_path):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
    domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]


def adaptation_method_lookup(adaptation):
    lookup_table = {"source": "Norm",
                    "norm_test": "Norm",
                    "norm_alpha": "Norm",
                    "norm_ema": "Norm",
                    "ttaug": "TTAug",
                    "memo": "MEMO",
                    "lame": "LAME",
                    "tent": "Tent",
                    "eata": "EATA",
                    "sar": "SAR",
                    "adacontrast": "AdaContrast",
                    "cotta": "CoTTA",
                    "rotta": "RoTTA",
                    "gtta": "GTTA",
                    "rmt": "RMT",
                    "roid": "ROID",
                    "proib": "Proib"
                    }
    assert adaptation in lookup_table.keys(), \
        f"Adaptation method '{adaptation}' is not supported! Choose from: {list(lookup_table.keys())}"
    return lookup_table[adaptation]