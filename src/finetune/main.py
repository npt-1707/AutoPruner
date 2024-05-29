from src.finetune.dataset import CallGraphDataset
from src.utils.utils import (
    Logger,
    AverageMeter,
    evaluation_metrics,
    read_config_file,
    load_json,
    save_json,
    set_seed,
)
from src.finetune.model import EmbeddingModel
from src.utils.loss_fn import get_loss_fn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import torch
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


def train(dataloader, model, mean_loss, loss_fn, optimizer, cfx_matrix):
    model.train()
    log_loss = []
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        code_ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        output, _ = model(code_ids, mask)

        loss = loss_fn(output, label)

        num_samples = output.shape[0]
        mean_loss.update(loss.item(), n=num_samples)

        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()

        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)

        if idx % 100 == 0:
            log_loss.append(mean_loss.item())
        loop.set_postfix(loss=mean_loss.item(), pre=precision, rec=recall, f1=f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, cfx_matrix, log_loss


def do_test(dataloader, model, logger):
    model.eval()
    cfx_matrix = np.array([[0, 0], [0, 0]])
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for batch, dl in loop:
        ids = dl["ids"].to(device)
        mask = dl["mask"].to(device)
        label = dl["label"].to(device)
        output, _ = model(ids, mask)

        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()

        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)
        loop.set_postfix(pre=precision, rec=recall, f1=f1)

    (tn, fp), (fn, tp) = cfx_matrix
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    logger.info("[EVAL] Precision {}, Recall {}, F1 {}".format(precision, recall, f1))
    return precision, recall, f1


def find_checkpoint(learned_model_dir):
    checkpoint_files = [
        f
        for f in os.listdir(learned_model_dir)
        if f.startswith("model_epoch_") and f.endswith(".pth")
    ]
    if len(checkpoint_files) == 0:
        return
    lastest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files])
    checkpoint = os.path.join(
        learned_model_dir, "model_epoch_{}.pth".format(lastest_epoch)
    )
    return checkpoint, lastest_epoch


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    f1 = checkpoint["max_f1"]
    return model, optimizer, f1


def do_train(
    epochs,
    train_loader,
    test_loader,
    model,
    loss_fn,
    optimizer,
    learned_model_dir,
    logger,
    loss_path,
):
    cfx_matrix = np.array([[0, 0], [0, 0]])
    mean_loss = AverageMeter()
    max_f1 = 0.0
    logs_loss = load_json(loss_path)
    last_epoch = -1
    checkpoint = find_checkpoint(learned_model_dir)
    if checkpoint is not None:
        checkpoint, last_epoch = checkpoint
        if last_epoch == epochs - 1:
            logger.info("Model has already been trained for {} epochs".format(epochs))
            return
        logger.info("Loaded checkpoint from {}".format(checkpoint))
        model, optimizer, max_f1 = load_checkpoint(checkpoint, model, optimizer)

    for epoch in range(last_epoch + 1, epochs):
        logger.info("Training at epoch {} ...".format(epoch))
        model, cfx_matrix, log_loss = train(
            train_loader, model, mean_loss, loss_fn, optimizer, cfx_matrix
        )
        logs_loss[epoch] = log_loss

        logger.info("Evaluating ...")
        _, _, f1 = do_test(test_loader, model, logger)
        if f1 > max_f1:
            max_f1 = f1
            logger.info("Saving best model ...")
            with open(os.path.join(learned_model_dir, "best_model.txt"), "w") as f:
                f.write("model_epoch_{}.pth".format(epoch))
        logger.info("Saving model ...")
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "max_f1": max_f1,
        }
        torch.save(
            state,
            os.path.join(learned_model_dir, "model_epoch_{}.pth".format(epoch)),
        )
        save_json(logs_loss, loss_path)
    logger.info("Finish training !!!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="codebert-base")
    parser.add_argument("--loss_fn", type=str, default="cross_entropy")
    parser.add_argument("--config_path", type=str, default="config/wala.config")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--train_batch_size", type=int, default=15)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="log")
    return parser.parse_args()


def main():
    args = get_args()
    config = read_config_file(args.config_path)

    # Logger
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_path = os.path.join(
        args.log_dir,
        "finetune_{}_{}_{}.log".format(args.model, args.loss_fn, args.mode),
    )
    logger = Logger(log_file=log_path)

    logger.info("Running on config {}".format(args.config_path))
    logger.info("Mode: {}".format(args.mode))

    # Dataset
    train_dataset = CallGraphDataset(config, "train", args.model, logger)
    test_dataset = CallGraphDataset(config, "test", args.model, logger)

    logger.info(
        "Dataset have {} train samples and {} test samples".format(
            len(train_dataset), len(test_dataset)
        )
    )
    TRAIN_PARAMS = {
        "batch_size": args.train_batch_size,
        "shuffle": True,
        "num_workers": 8,
    }
    TEST_PARAMS = {
        "batch_size": args.test_batch_size,
        "shuffle": False,
        "num_workers": 8,
    }

    train_loader = DataLoader(train_dataset, **TRAIN_PARAMS)
    test_loader = DataLoader(test_dataset, **TEST_PARAMS)

    # Model, loss function, optimizer
    model = EmbeddingModel(args.model)

    if torch.cuda.device_count() > 1:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    loss_fn = get_loss_fn(args.loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # checkpoint directory
    learned_model_dir = config["LEARNED_MODEL_DIR"]
    learned_model_dir = os.path.join(learned_model_dir, args.model, args.loss_fn)
    if not os.path.exists(learned_model_dir):
        os.makedirs(learned_model_dir)

    # train/test
    if args.mode == "train":
        loss_path = os.path.join(
            learned_model_dir, f"log_loss_{args.model}_{args.loss_fn}.json"
        )
        do_train(
            args.epochs,
            train_loader,
            test_loader,
            model,
            loss_fn,
            optimizer,
            learned_model_dir,
            logger,
            loss_path,
        )
    elif args.mode == "test":
        with open(os.path.join(learned_model_dir, "best_model.txt"), "r") as f:
            best_model_path = f.read().strip()
            best_model_path = os.path.join(learned_model_dir, best_model_path)
        load_checkpoint(best_model_path, model, optimizer)
        do_test(test_loader, model, logger)
    else:
        raise NotImplemented


if __name__ == "__main__":
    main()
