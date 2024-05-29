from src.training.dataset import FinetunedDataset
from src.gnn.dataset import CallGraphDataset
from src.gnn.model import GCNModel
from src.training.model import NNClassifier_Combine
from src.utils.utils import read_config_file, Logger
from src.finetune.model import get_emb_size
import math
import numpy as np
import os
import torch
import statistics
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score

TEST_PARAMS = {"batch_size": 100, "shuffle": False, "num_workers": 8}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config")

    parser.add_argument("--model", type=str, default="codebert")
    parser.add_argument("--loss_fn", type=str, default="cross_entropy")
    parser.add_argument("--ensemble", type=str, default="max")
    return parser.parse_args()


def do_test(logger, test_loader, model, test_g_loader, g_model, ensemble="max"):
    model.eval()
    cfx_matrix = np.array([[0, 0], [0, 0]])
    result_per_programs = {}
    for i in range(41):
        result_per_programs[i] = {"lb": [], "output": [], "g_output": []}

    for i, batch in tqdm(enumerate(test_loader)):
        code = batch["code"].to(device)
        struct = batch["struct"].to(device)
        label = batch["label"].to(device)
        sanity_check = batch["static"].numpy()
        program_ids = batch["program_ids"].numpy()
        output = model(code=code, struct=struct)
        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        output = output * sanity_check
        label = label.detach().cpu().numpy()

        for i in range(len(label)):
            prog_idx, out, lb = program_ids[i], output[i], label[i]
            result_per_programs[prog_idx]["lb"].append(lb)
            result_per_programs[prog_idx]["output"].append(out)

    g_model.eval()
    loop = tqdm(enumerate(test_g_loader), leave=False, total=len(test_g_loader))
    for idx, batch in loop:
        g, lb, sa_lb = batch
        g = g.to(device)
        lb = lb.to(device)
        sa_lb = sa_lb
        g = g_model(g)

        output = g.edata["prob"]
        output = output.detach().cpu().numpy()[:, 1]
        output = output * sa_lb
        lb = lb.detach().cpu().numpy()
        assert (
            (lb == result_per_programs[idx]["lb"]).all()
        ), f"Mismatch labels in program ids {idx}"
        result_per_programs[idx]["g_output"] = output

    precisions, recalls, f1s = [], [], []
    for i in range(41):
        output = result_per_programs[i]["output"]
        g_output = result_per_programs[i]["g_output"]
        if ensemble == "max":
            pred = np.max([output, g_output], axis=0)
        elif ensemble == "avg":
            pred = np.mean([output, g_output], axis=0)
        else:
            raise NotImplemented
        pred = np.where(pred >= 0.5, 1, 0)
        label = result_per_programs[i]["lb"]
        p = precision_score(label, pred, labels=[0, 1])
        if math.isnan(p):
            p = 0
        r = recall_score(label, pred, labels=[0, 1])
        if p + r != 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    logger.log(
        "[EVAL-AVG] Precision {} ({}), Recall {} ({}), F1 {} ({})".format(
            round(statistics.mean(precisions), 2),
            round(statistics.stdev(precisions), 2),
            round(statistics.mean(recalls), 2),
            round(statistics.stdev(recalls), 2),
            round(statistics.mean(f1s), 2),
            round(statistics.stdev(f1s), 2),
        )
    )


def main():
    args = get_args()
    config = read_config_file(args.config_path)
    logger = Logger()
    
    test_ft_dataset = FinetunedDataset(config, "test", args.model, args.loss_fn)
    test_ft_loader = DataLoader(test_ft_dataset, **TEST_PARAMS)
    model = NNClassifier_Combine(input_size=get_emb_size(args.model), hidden_size=32)
    model.to(device)
    model_checkpoint = os.path.join(
        config["CLASSIFIER_MODEL_DIR"], args.model, args.loss_fn, "model.pth"
    )
    model.load_state_dict(torch.load(model_checkpoint))
    
    test_g_loader = CallGraphDataset(config, "test")
    g_model = GCNModel(config, 32)
    g_model.to(device)
    g_model_checkpoint = os.path.join(config["GNN_MODEL_DIR"], "gnn_wala.pth")
    g_model.load_state_dict(torch.load(g_model_checkpoint))

    do_test(logger, test_ft_loader, model, test_g_loader, g_model, args.ensemble)
if __name__ == "__main__":
    main()
