from src.training.dataset import FinetunedDataset
from torch.utils.data import DataLoader
from src.training.model import (
    NNClassifier_Combine,
    NNClassifier_Structure,
    NNClassifier_Semantic,
)
from src.utils.utils import (
    Logger,
    AverageMeter,
    evaluation_metrics,
    read_config_file,
    load_json,
    save_json,
    set_seed,
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
import argparse
import os
import warnings
import torch
import math
import statistics

warnings.filterwarnings("ignore")

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PARAMS = {"batch_size": 100, "shuffle": True, "num_workers": 8}
TEST_PARAMS = {"batch_size": 100, "shuffle": False, "num_workers": 8}


def train(dataloader, model, mean_loss, loss_fn, optimizer, cfx_matrix):
    model.train()
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    log_loss = []
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

        # logger.info("Iter {}: Loss {}, Precision {}, Recall {}, F1 {}".format(idx, loss.item(), precision, recall, f1))
        if idx % 500 == 0:
            log_loss.append(mean_loss.item())
        loop.set_postfix(loss=mean_loss.item(), pre=precision, rec=recall, f1=f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, cfx_matrix, log_loss


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
    return statistics.mean(f1_avg)


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
    for epoch in range(epochs):
        logger.info("Start training at epoch {} ...".format(epoch))
        model, cfx_matrix, log_loss = train(
            train_loader, model, mean_loss, loss_fn, optimizer, cfx_matrix
        )
        logs_loss[epoch] = log_loss

        logger.info("Saving model ...")
        torch.save(
            model.state_dict(),
            os.path.join(learned_model_dir, f"model_epoch_{epoch}.pth"),
        )

        logger.info("Evaluating ...")
        f1 = do_test(test_loader, model, logger, False)
        if f1 > max_f1:
            max_f1 = f1
            logger.info("Saving best model ...")
            torch.save(model.state_dict(), os.path.join(learned_model_dir, "model.pth"))
    save_json(logs_loss, loss_path)
    logger.info("Done !!!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument(
        "--feature", type=int, default=2, help="0: structure, 1: semantic, 2:combine"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", type=str, default="codebert-base")
    parser.add_argument("--loss_fn", type=str, default="cross_entropy")
    parser.add_argument("--log_dir", type=str, default="log")

    return parser.parse_args()


def main():
    args = get_args()
    config = read_config_file(args.config_path)

    # Logger
    log_path = os.path.join(
        args.log_dir, f"train_{args.model}_{args.loss_fn}_{args.mode}.log"
    )
    logger = Logger(log_path)

    logger.info("Running on config {}".format(args.config_path))
    logger.info("Mode: {}".format(args.mode))

    mode = args.mode

    learned_model_dir = config["CLASSIFIER_MODEL_DIR"]
    learned_model_dir = os.path.join(learned_model_dir, args.model, args.loss_fn)
    if not os.path.exists(learned_model_dir):
        os.makedirs(learned_model_dir)

    train_dataset = FinetunedDataset(config, "train", args.model, args.loss_fn, logger)
    test_dataset = FinetunedDataset(config, "test", args.model, args.loss_fn, logger)

    logger.info(
        "Dataset have {} train samples and {} test samples".format(
            len(train_dataset), len(test_dataset)
        )
    )

    train_loader = DataLoader(train_dataset, **TRAIN_PARAMS)
    test_loader = DataLoader(test_dataset, **TEST_PARAMS)

    if args.feature == 2:
        input_size = models[args.model]["embedding_size"]
        model = NNClassifier_Combine(input_size=input_size, hidden_size=32)
    elif args.feature == 1:
        input_size = models[args.model]["embedding_size"]
        model = NNClassifier_Semantic(input_size=input_size, hidden_size=32)
    elif args.feature == 0:
        model = NNClassifier_Structure(32)
    else:
        raise NotImplemented

    if torch.cuda.device_count() > 1:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)

    model.apply(init_weights)

    loss_fn = get_loss_fn(args.loss_fn)

    optimizer = optim.Adam(model.parameters(), lr=5e-6)

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
        model.load_state_dict(torch.load(os.path.join(learned_model_dir, "model.pth")))
        do_test(test_loader, model, logger, True)
    else:
        raise NotImplemented


if __name__ == "__main__":
    main()

    # a = [0, 1, 0, 0, 1, 0]
    # b = [0, 1, 0, 1, 0, 1]
    # fpr, tpr, thresholds = roc_curve(a, b)
    # logger.info(roc_curve(a, b, pos_label=2))
