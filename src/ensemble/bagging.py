from src.training.dataset import FinetunedDataset
from torch.utils.data import DataLoader
from src.training.model import NNClassifier_Combine
from src.ensemble.model import BaggingModel
from src.finetune.model import models
from src.utils.utils import (
    Logger,
    AverageMeter,
    evaluation_metrics,
    read_config_file,
    load_json,
    save_json,
)
from src.utils.loss_fn import get_loss_fn
from src.finetune.model import models
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score
import numpy as np
import torch.optim as optim
import torch
import math
import statistics
from argparse import ArgumentParser
import warnings
import os

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataloader, model, mean_loss, loss_fn, optimizer, cfx_matrix):
    model.train()
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        code = batch["code"].to(device)
        struct = batch["struct"].to(device)
        label = batch["label"].to(device)
        output = model(code=code, struct=struct)

        loss = loss_fn(output, label)
        # logger.info(output)
        # logger.info(label)
        num_samples = output.shape[0]
        mean_loss.update(loss.item(), n=num_samples)

        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()

        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)
        loop.set_postfix(loss=mean_loss.item(), pre=precision, rec=recall, f1=f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, cfx_matrix


def do_test(dataloader, model, logger, is_write=False):
    model.eval()
    cfx_matrix = np.array([[0, 0], [0, 0]])
    result_per_programs = {}
    for i in range(41):
        result_per_programs[i] = {"lb": [], "output": []}

    all_outputs = []
    all_labels = []
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        code = batch["code"].to(device)
        struct = batch["struct"].to(device)
        label = batch["label"].to(device)
        sanity_check = batch["static"].numpy()
        program_ids = batch["program_ids"].numpy()
        output = model(code=code, struct=struct)
        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        output = output * sanity_check
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()

        for i in range(len(label)):
            prog_idx, out, lb = program_ids[i], output[i], label[i]
            result_per_programs[prog_idx]["lb"].append(lb)
            result_per_programs[prog_idx]["output"].append(out)
            all_outputs.append(out)
            all_labels.append(lb)

        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)
        loop.set_postfix(pre=precision, rec=recall, f1=f1)

    if is_write:
        np.save("prediction.npy", np.array(all_outputs))

    (tn, fp), (fn, tp) = cfx_matrix
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    logger.info(
        "[EVAL] Iter {}, Precision {}, Recall {}, F1 {}".format(
            idx, precision, recall, f1
        )
    )

    precision_avg, recall_avg, f1_avg = [], [], []
    for i in range(41):
        lb = np.array(result_per_programs[i]["lb"])
        output = np.array(result_per_programs[i]["output"])
        pred = np.where(output >= 0.5, 1, 0)
        temp = precision_score(lb, pred), recall_score(lb, pred)
        if math.isnan(temp[0]):
            temp[0] = 0
        precision_avg.append(temp[0])
        recall_avg.append(temp[1])
        if temp[0] + temp[1] != 0:
            f1_avg.append(2 * temp[0] * temp[1] / (temp[0] + temp[1]))
        else:
            f1_avg.append(0)
    logger.info(
        "[EVAL-AVG] Iter {}, Precision {} ({}), Recall {} ({}), F1 {} ({})".format(
            idx,
            round(statistics.mean(precision_avg), 2),
            round(statistics.stdev(precision_avg), 2),
            round(statistics.mean(recall_avg), 2),
            round(statistics.stdev(recall_avg), 2),
            round(statistics.mean(f1_avg), 2),
            round(statistics.stdev(f1_avg), 2),
        )
    )


def do_train(
    epochs,
    train_loader,
    test_loader,
    bagging_model,
    loss_fn,
    logger,
    learned_model_dir,
):
    for idx, model in enumerate(bagging_model.estimators):
        logger.info(f"Training estimator {idx} ...")
        cfx_matrix = np.array([[0, 0], [0, 0]])
        mean_loss = AverageMeter()
        optimizer = bagging_model.optimizers[idx]
        for epoch in range(epochs):
            logger.info("Start training at epoch {} ...".format(epoch))
            model, cfx_matrix = train(
                train_loader, model, mean_loss, loss_fn, optimizer, cfx_matrix
            )
            logger.info("Evaluating ...")
            do_test(test_loader, model, logger)
    logger.info("Saving model ...")
    save_path = os.path.join(learned_model_dir, f"bagging_model_{len(bagging_model)}_estimators.pth")
    torch.save(bagging_model.state_dict(), save_path)
    logger.info("Done !!!")


def load_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config")
    parser.add_argument("--model", type=str, default="codebert-base")
    parser.add_argument("--loss_fn", type=str, default="cross_entropy")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)
    # parser.add_argument("--log_iter", type=int, default=250)
    parser.add_argument("--num_est", type=int, default=5)
    parser.add_argument("--mode", type=str, default="train")
    return parser.parse_args()


def main():
    args = load_args()
    config = read_config_file(args.config_path)
    log_dir = os.path.join(args.log_dir, "bagging")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(
        log_dir, "bagging_{}_{}.log".format(args.model, args.loss_fn)
    )
    logger = Logger(log_path)
    logger.info(f"Start training BaggingModel<{args.model}>")
    learned_model_dir = config["CLASSIFIER_MODEL_DIR"]
    learned_model_dir = os.path.join(
        learned_model_dir, args.model, args.loss_fn, "bagging"
    )
    if not os.path.exists(learned_model_dir):
        os.makedirs(learned_model_dir)

    train_dataset = FinetunedDataset(config, "train", args.model, args.loss_fn, logger)
    test_dataset = FinetunedDataset(config, "test", args.model, args.loss_fn, logger)
    logger.info(
        "Dataset have {} train samples and {} test samples".format(
            len(train_dataset), len(test_dataset)
        )
    )
    TRAIN_PARAMS = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 4,
    }
    TEST_PARAMS = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 4,
    }

    train_loader = DataLoader(train_dataset, **TRAIN_PARAMS)
    test_loader = DataLoader(test_dataset, **TEST_PARAMS)

    model = BaggingModel(
        estimator=NNClassifier_Combine,
        n_estimators=args.num_est,
        estimator_params={
            "hidden_size": 32,
            "input_size": models[args.model]["embedding_size"],
        },
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    model.set_adam_optimizer(lr=5e-6)
    if args.mode == "train":
        do_train(
            args.epoch,
            train_loader,
            test_loader,
            model,
            criterion,
            logger,
            learned_model_dir,
        )
    elif args.mode == "test":
        logger.info(f"Evaluating BaggingModel<{args.model}> ...")
        save_path = os.path.join(learned_model_dir, f"bagging_model_{args.num_est}_estimators.pth")
        model.load_state_dict(torch.load(save_path))
        pred_path = "output"
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        pred_path = os.path.join(pred_path, f"pred_bagging_{args.model}_{args.loss_fn}.npy")
        do_test(test_loader, model, logger, pred_path)
    else:
        raise NotImplemented


if __name__ == "__main__":
    main()
