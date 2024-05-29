from src.finetune.dataset import CallGraphDataset
from src.utils.utils import read_config_file, Logger, set_seed
from argparse import ArgumentParser
import os

set_seed(42)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="codebert-base")
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--log_dir", type=str, default="log")
    return parser.parse_args()

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    log_path = os.path.join(args.log_dir, "preload")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, "preload_for_finetune_{}_{}.log".format(args.model, args.mode))
    logger = Logger(log_path)
    CallGraphDataset(config, args.mode, args.model, logger)

if __name__ == "__main__":
    main()