import os
from omegaconf import OmegaConf

from common.register import registry
from dataset import *
from models import *
from runners import *
from utils.get_args import get_args
# from utils.logger import get_color_logger
from utils.set_seed import set_seed
import datetime
from torch import multiprocessing as mp


def read_config(config_file, args):
    
    assert os.path.isfile(config_file), f"config file {config_file} doesn't eixst!"
    cfg = OmegaConf.load(config_file)
    
    cfg.run.update({"save_dir": args.save_dir})

    return cfg

def main():

    args = get_args()
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.save_dir = os.path.join(args.save_dir, time)

    log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    set_seed(args.seed)

    cfg = read_config(args.config_file, args=args)
    cfg.run.update({"log_dir": log_dir})

    if args.distribute:
        devices = [int(item) for item in args.device.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        assert len(devices) > 1, "distribute training must have less 2 gpus!"
        world_size = len(devices)
        try:
            mp.spawn(
                registry.get_runner_class(cfg.run.get("arch", "BaseTrainer")).from_config,
                nprocs=world_size,
                args=(world_size, args.distribute, cfg),
                join=True
            )
        finally:
            torch.distributed.destroy_process_group()
    else:
        trainer = registry.get_runner_class(cfg.run.get("arch", "BaseTrainer")).from_config(cfg=cfg)

if __name__ == "__main__":
    main()
