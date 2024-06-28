import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import nn
from src.finetune.dataset import CallGraphDataset
from src.utils.utils import read_config_file, Logger, set_seed
from src.finetune.model import EmbeddingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

def get_memory_usage(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        return 0

def save_finetune(config, mode, model_name, loss_fn, logger, batch_size=10):
    logger.info(f"Saving finetuned embeddings for {model_name} model")
    save_dir = os.path.join(
        config["CACHE_DIR"],
        model_name,
        loss_fn,
        "{}_finetuned_{}".format(mode, batch_size),
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(os.listdir(save_dir)) > 0:
        logger.info("Directory {} already exists".format(save_dir))
        return
    PARAMS = {"batch_size": batch_size, "shuffle": False, "num_workers": 4}
    model_path = os.path.join(
        config["LEARNED_MODEL_DIR"],
        model_name,
        loss_fn,
    )
    model_path = os.path.join(model_path, "model_epoch_1.pth")
    dataset = CallGraphDataset(config, mode, model_name, logger)
    dataloader = DataLoader(dataset, **PARAMS)
    model = EmbeddingModel(model_name)

    # if torch.cuda.device_count() > 1:
    #     logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        if mode == "test":
            logger.info(f"Memory used: {get_memory_usage(device)}")
        _, emb = model(ids=ids, mask=mask)
        if mode == 'test':
            logger.info(f"Memory used: {get_memory_usage(device)}")
        emb = emb.detach().cpu().numpy()
        save_path = os.path.join(save_dir, "{}.npy".format(idx))
        np.save(save_path, emb)
    logger.info(f"Saved into {save_dir}!")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="config/wala.config", required=True
    )
    parser.add_argument("--model", type=str, default="codebert-base", required=True)
    parser.add_argument("--loss_fn", type=str, default="cross_entropy", required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="log")
    return parser.parse_args()


def main():
    args = get_args()
    config = read_config_file(args.config_path)
    log_path = os.path.join(args.log_dir, "save_finetune")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, "save_finetune_{}.log".format(args.model))
    logger = Logger(log_path)
    # save_finetune(config, "train", args.model, args.loss_fn, logger, args.batch_size)
    save_finetune(config, "test", args.model, args.loss_fn, logger, args.batch_size)


if __name__ == "__main__":
    main()
