# src/train/train_wnn.py
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

import numpy as np
from os.path import join

from src.core.infer import build_runtime_profile
from src.dataio.encode import encode_batch_thermo_plus_sobel

from datasets.LoadDatasets import MnistDataloader
from src.tools.utils import _addr_from_bits


# ---------------------------
# Predict / Eval with profile
# ---------------------------

def predict_with_profile(profile: Dict, bit_vec: np.ndarray) -> int:
    lut_table = profile["lut_table"]                     # (C, L, A)
    kept_bits_per_lut = profile["kept_global_bits_per_lut"]
    alpha = profile["alpha"]
    C, L, A = lut_table.shape

    votes = np.zeros((C, L), dtype=np.float32)
    for l in range(L):
        addr = _addr_from_bits(bit_vec, kept_bits_per_lut[l])
        votes[:, l] = lut_table[:, l, addr]

    denom = votes.sum(axis=0, keepdims=True) + C * alpha
    post = (votes + alpha) / denom
    scores = np.log(post + 1e-9).sum(axis=1)            # baseline log-ratio
    return int(np.argmax(scores))


def eval_with_profile(profile: Dict, X_bits: np.ndarray, y: np.ndarray) -> float:
    X_bits = np.asarray(X_bits)
    y = np.asarray(y)
    correct = 0
    for i in range(X_bits.shape[0]):
        pred = predict_with_profile(profile, X_bits[i])
        if pred == int(y[i]):
            correct += 1
    return correct / X_bits.shape[0]


# ---------------------------
# Grid evaluation (bits_keep × luts_keep)
# ---------------------------
def eval_grid_bits_luts(model,
                        tuple_mapping: List[List[int]],
                        bit_priority: np.ndarray,
                        lut_priority: np.ndarray,
                        X_bits: np.ndarray,
                        y: np.ndarray,
                        bits_keep_list: List[float],
                        luts_keep_list: List[float],
                        *,
                        align_strategy: str = "min") -> Dict[Tuple[float, float], float]:
    """
    return {(bits_keep, luts_keep): acc}
    """
    from collections import OrderedDict
    results = OrderedDict()
    all_bits = set(range(X_bits.shape[1]))

    for bk in bits_keep_list:
        # build global set
        B = bit_priority.shape[0]
        order_bits = np.argsort(-bit_priority)
        kbits = max(1, int(round(B * bk)))
        keep_bits_set = set(order_bits[:kbits])

        # bit pruning sanity check：when bk==1.0, keep_bits_set == all_bits
        if abs(bk - 1.0) < 1e-9:
            keep_bits_set = all_bits

        for lk in luts_keep_list:
            profile = build_runtime_profile(
                model=model,
                tuple_mapping=tuple_mapping,
                keep_bits_set=keep_bits_set,
                lut_priority=lut_priority,
                luts_keep_ratio=lk,
                bit_priority=bit_priority,
                align_strategy=align_strategy
            )
            acc = eval_with_profile(profile, X_bits, y)
            results[(bk, lk)] = acc
    return results


# ---------------------------
# Sanity: Full-full consistency check
# ---------------------------
def sanity_check_full_profile(model,
                              tuple_mapping: List[List[int]],
                              X_bits: np.ndarray,
                              y: np.ndarray,
                              lut_priority: np.ndarray) -> float:
    all_bits = set(range(X_bits.shape[1]))
    profile = build_runtime_profile(
        model=model,
        tuple_mapping=tuple_mapping,
        keep_bits_set=all_bits,
        lut_priority=lut_priority,
        luts_keep_ratio=1.0,
        align_strategy="min"  # target_m = n
    )
    acc = eval_with_profile(profile, X_bits, y)
    return acc


def predict_with_profile_varm(profile: Dict, bit_vec: np.ndarray, mode: str = "log_posterior") -> int:
    alpha = profile["alpha"]
    C = profile["num_classes"]
    scores = np.zeros(C, dtype=np.float32)

    for lut_tab, gbits in zip(profile["lut_tables"], profile["kept_global_bits_per_lut"]):
        addr = _addr_from_bits(bit_vec, gbits)
        counts = lut_tab[:, addr]  # (C,)
        if mode == "log_count":
            scores += np.log(counts + alpha + 1e-9)
        else:
            denom = counts.sum() + C * alpha
            post = (counts + alpha) / denom
            log_post = np.log(post + 1e-9)
            if mode == "zero_mean_log_posterior":
                log_post = log_post - log_post.mean()
            scores += log_post

    return int(np.argmax(scores))

##############################
#
##############################
@torch.no_grad()
def eval_acc(model, data_loader, device):
    model.eval()
    total = 0
    correct = 0
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        total += yb.numel()
        correct += (preds == yb).sum().item()
    return correct / total


@torch.no_grad()
def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)  
        loss = F.cross_entropy(logits, yb)

        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += yb.numel()
        total_loss += loss.item() * yb.numel()

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


if __name__ == "__main__":
    # load dataset
    input_path = 'D:/workspace/Adaptive_WNN/datasets'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (_, _), (x_test, y_test) = mnist_dataloader.load_data()

    x_test_bit_Vec, _ = encode_batch_thermo_plus_sobel(
        x_test, tiles=(4, 4), levels=8, sobel_threshold_ratio=0.2
    )

    random.seed(time.time())
    seed = random.randint(0, 100)



