from collections import defaultdict
from typing import List, Tuple, Dict
from src.tools.utils import _assert_power_of_two
from src.dataio.encode import bucket_mapper_mnist_thermo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_keep_mask(w_lut, keep_ratio):
    L = w_lut.shape[0]
    order = np.argsort(-w_lut)  # desc
    k = max(1, int(round(L * keep_ratio)))
    keep_ids = order[:k]
    mask = np.zeros(L, dtype=np.float32)
    mask[keep_ids] = 1.0
    return mask, keep_ids.tolist()  # shape (L,)

def eval_masked(model, X_bits, y, lut_mask, alpha: float = 1.0):
    correct = 0
    for i in range(X_bits.shape[0]):
        pred = predict_masked(model, X_bits[i], lut_mask, alpha=alpha)
        if pred == int(y[i]):
            correct += 1
    return correct / X_bits.shape[0]


def predict_masked(model, bit_vec, lut_mask, alpha: float = 1.0):
    """
    baseline log-ratio scoring，only use the LUT's lut_mask==1。
    lut_mask: shape (L,), values in {0.0, 1.0}
    """
    C = model.num_classes
    L = model.num_luts_per_class
    assert lut_mask.shape[0] == L

    addr = model._addresses_for_sample(bit_vec)  # (L,)

    votes = model.table[
        np.arange(C)[:, None],
        np.arange(L)[None, :],
        addr[None, :]
    ].astype(np.float32)  # (C,L)

    denom = votes.sum(axis=0, keepdims=True) + C * alpha  # (1,L)
    post = (votes + alpha) / denom  # (C,L)
    log_post = np.log(post + 1e-9)  # (C,L)

    # set contribution of the pruned LUT to 0
    masked_log = log_post * lut_mask[None, :]  # (C,L)

    scores = masked_log.sum(axis=1)  # (C,)
    return int(np.argmax(scores))

def drop_luts_by_priority(profile, lut_priority, keep_ratio):
    keep = max(1, int(round(len(profile["lut_tables"]) * keep_ratio)))
    order = np.argsort(-lut_priority)[:keep]  # keep first k

    lut_tables = [profile["lut_tables"][i] for i in order]
    kept_bits  = [profile["kept_global_bits_per_lut"][i] for i in order]
    m_list     = [profile["addr_bits_per_lut"][i] for i in order]
    return dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_bits,
        addr_bits_per_lut=m_list,
        num_classes=profile["num_classes"],
        alpha=profile["alpha"],
        bit_order="lsb",
    )


def select_top_luts_by_priority(lut_priority: np.ndarray, keep_ratio: float) -> np.ndarray:
    """based on LUT's priority, keep top-K%"""
    L = len(lut_priority)
    k = max(1, int(round(L * keep_ratio)))
    order = np.argsort(-lut_priority)[:k]
    return np.sort(order)

def prune_model_by_lut_indices(model, tuple_mapping: List[List[int]], keep_ids: np.ndarray):
    """prune original model and mapping table"""
    keep_ids = np.asarray(keep_ids, dtype=int)
    new_table = model.table[:, keep_ids, :].copy()        # (C, L_kept, 2^n)
    new_mapping = [tuple_mapping[i] for i in keep_ids]    # length L_kept
    return new_table, new_mapping

def drop_profile_to_luts(profile: Dict, keep_ids: np.ndarray) -> Dict:
    """from profile（bit-pruned） do the furthur LUT pruning。"""
    keep_ids = np.asarray(keep_ids, dtype=int)
    lut_tables = [profile["lut_tables"][i] for i in keep_ids]
    kept_bits  = [profile["kept_global_bits_per_lut"][i] for i in keep_ids]
    m_list     = [profile["addr_bits_per_lut"][i] for i in keep_ids]
    return dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_bits,
        addr_bits_per_lut=m_list,
        num_classes=profile["num_classes"],
        alpha=profile["alpha"],
        bit_order=profile.get("bit_order","lsb"),
        meta=[profile.get("meta", [{}]*len(profile["lut_tables"]))[i] for i in keep_ids]
    )

def _pick_local_bits(
    mode: str,
    lut_bits: List[int],
    bit_priority: np.ndarray,
    k: int,
    coverage_r: int,
    bucket_mapper,
    coverage_k_threshold: int,
) -> List[int]:
    if mode == "hard":
        return select_local_bits_with_coverage(
            lut_bits, bit_priority, k,
            r_per_bucket=coverage_r, bucket_mapper=bucket_mapper
        )
    else:
        return select_local_bits_soft_coverage(
            lut_bits, bit_priority, k,
            r_per_bucket_when_small=coverage_r,
            bucket_mapper=bucket_mapper,
            coverage_k_threshold=coverage_k_threshold
        )



def compute_bit_priority_entropy(X_bits: np.ndarray, y: np.ndarray, num_classes: int, eps: float = 1e-9):
    """
    X_bits: (N, B) 0/1 bit matrix
    y:      (N,)   標籤 0..C-1
    num_classes: C
    return:
      bit_priority: (B,) larger -> more important (priority = -entropy)
    """
    N, B = X_bits.shape
    C = num_classes

    y = np.array(y)
    X_bits = np.array(X_bits)

    # class × bit
    ones_cb = np.zeros((C, B), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if np.any(idx):
            ones_cb[c] = X_bits[idx].sum(axis=0)

    # #samples
    class_cnt = np.array([np.sum(y == c) for c in range(C)], dtype=np.float64) + eps  # (C,)
    # p(bit=1 | class=c)
    p1_cb = ones_cb / class_cnt[:, None]  # (C,B)

    # w_c ∝ p1_cb[c,b] * P(class=c)
    p_class = class_cnt / class_cnt.sum()
    w_cb = p1_cb * p_class[:, None]           # (C,B)
    w_sum = w_cb.sum(axis=0, keepdims=True) + eps
    p_class_given_bit = w_cb / w_sum          # (C,B)

    # entropy over classes for each bit
    entropy_b = -np.sum(p_class_given_bit * np.log(p_class_given_bit + eps), axis=0)  # (B,)
    priority_b = -entropy_b.astype(np.float32)
    return priority_b


# ---------------------------
# Core: reduce one LUT by kept local positions
# ---------------------------
def _reduce_counts_for_lut(counts_c_a: np.ndarray,
                           keep_positions_in_lut: List[int],
                           n_addr_bits: int) -> Tuple[np.ndarray, List[int]]:
    """
    counts_c_a: (C, 2^n)
    keep_positions_in_lut
    return:
      reduced: (C, 2^m)
      kept_local_pos_sorted: List[int], asc
    """
    counts_c_a = np.asarray(counts_c_a)
    C, A = counts_c_a.shape
    _assert_power_of_two(A)
    n_from_A = A.bit_length() - 1
    if n_addr_bits != n_from_A:
        n_addr_bits = n_from_A

    keep_positions = sorted(set(int(p) for p in keep_positions_in_lut))
    if any(p < 0 or p >= n_addr_bits for p in keep_positions):
        raise ValueError(f"keep_positions {keep_positions} out of range [0,{n_addr_bits-1}]")

    drop_positions = [p for p in range(n_addr_bits) if p not in keep_positions]

    # (C, 2^n) -> (C, [2]*n)
    arr = counts_c_a.reshape(C, *([2] * n_addr_bits))
    if drop_positions:
        axes_to_sum = tuple(1 + p for p in sorted(drop_positions))
        arr = arr.sum(axis=axes_to_sum, keepdims=False)

    reduced = arr.reshape(C, -1)  # (C, 2^m)
    return reduced, keep_positions


# ------------------  per-LUT select top-k by coverage per LUT ------------------
def select_local_bits_with_coverage(lut_bits: List[int],
                                    bit_priority: np.ndarray,
                                    k: int,
                                    r_per_bucket: int = 1,
                                    bucket_mapper=bucket_mapper_mnist_thermo) -> List[int]:
    """
    return：local addr index
    """
    n = len(lut_bits)
    k = max(1, min(k, n))

    # order by bit_priority
    local_scored = [(p, float(bit_priority[lut_bits[p]])) for p in range(n)]
    local_scored.sort(key=lambda x: x[1], reverse=True)

    # distribution
    buckets = defaultdict(list)  # name -> list[(pos, score)]
    for p, sc in local_scored:
        name = bucket_mapper(lut_bits[p])
        buckets[name].append((p, sc))

    # satisfy the coverage
    keep = []
    if r_per_bucket > 0:
        for name, items in buckets.items():
            take = min(r_per_bucket, len(items))
            keep.extend([pos for pos, _ in items[:take]])

    # make sure the #keeps reach k
    if len(keep) < k:
        taken = set(keep)
        remaining = [it for sub in buckets.values() for it in sub]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for pos, _ in remaining:
            if pos not in taken:
                keep.append(pos); taken.add(pos)
                if len(keep) == k: break

    keep = sorted(set(keep))
    if len(keep) == 0:
        keep = [local_scored[0][0]]
    return keep



def profile_stats(profile, n_full):
    # n_full = the bitwidth during training
    m_list = profile["addr_bits_per_lut"]          # each LUT's m_l
    L_kept = len(m_list)
    avg_m = sum(m_list) / L_kept
    entries = sum(1<<m for m in m_list)
    entries_full = L_kept * (1<<n_full)
    compression = entries / entries_full           # < 1.0 small is the better
    return dict(L=L_kept, avg_m=avg_m, entries=entries,
                compression=compression)

def select_local_bits_soft_coverage(
    lut_bits: List[int],
    bit_priority: np.ndarray,
    k: int,
    *,
    r_per_bucket_when_small: int = 1,
    bucket_mapper=bucket_mapper_mnist_thermo,
    coverage_k_threshold: int = 4
) -> List[int]:
    """
    Soft coverage:
      - If k <= coverage_k_threshold: use coverage (per bucket keep r).
      - Else: pure per-LUT top-k by bit_priority (no coverage).
    """
    n = len(lut_bits)
    k = max(1, min(k, n))
    # rank locally by bit_priority
    local_scored = [(p, float(bit_priority[lut_bits[p]])) for p in range(n)]
    local_scored.sort(key=lambda x: x[1], reverse=True)

    if k <= coverage_k_threshold and r_per_bucket_when_small > 0:
        # coverage phase (same as your hard version)
        from collections import defaultdict
        buckets = defaultdict(list)
        for p, sc in local_scored:
            buckets[bucket_mapper(lut_bits[p])].append((p, sc))

        keep = []
        for _, items in buckets.items():
            items.sort(key=lambda x: x[1], reverse=True)
            take = min(r_per_bucket_when_small, len(items))
            keep.extend([pos for pos, _ in items[:take]])

        if len(keep) < k:
            taken = set(keep)
            remaining = [it for sub in buckets.values() for it in sub]
            remaining.sort(key=lambda x: x[1], reverse=True)
            for pos, _ in remaining:
                if pos not in taken:
                    keep.append(pos); taken.add(pos)
                    if len(keep) == k: break
        keep = sorted(set(keep))
        if not keep:
            keep = [local_scored[0][0]]
        return keep
    else:
        # non-coverage phase (free to compress)
        return sorted([p for p, _ in local_scored[:k]])
    


##############################
#
#####################3########
@torch.no_grad()
def collect_hidden_activations(model, data_loader, device):
    """
    collect the last hidden h_last
    return:
      H: [N, H]  (put all the sample into a table)
      Y: [N]     (corresponding label)
    """
    model.eval()
    all_h = []
    all_y = []

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, h_last = model(xb, return_hidden=True)  # h_last: [B, H]
        all_h.append(h_last.cpu())
        all_y.append(yb.cpu())

    H = torch.cat(all_h, dim=0)  # [N, H]
    Y = torch.cat(all_y, dim=0)  # [N]
    return H, Y


def compute_importance_weighted(H: torch.Tensor, model: nn.Module):
    """
    H: [N, H] last hidden's activation
    model: pretrained WNN (includes classifier)

    importance_j = std(h_j) * sum_c |W[c,j]|
    """
    with torch.no_grad():
        std = H.std(dim=0)                    # [H]
        W = model.classifier.weight.data      # [C, H]
        w_abs = W.abs().sum(dim=0)            # [H]
        importance = std * w_abs
    return importance

def build_pruned_classifier(model, importance, keep_ratio=0.5, min_keep=64):
    """
    according to the importance, select the keeped hidden dimension
    create classifier, and put the keep_idx back model
    """
    device = next(model.parameters()).device
    H = importance.numel()

    keep_dim = max(min_keep, int(H * keep_ratio))
    keep_dim = min(keep_dim, H)

    _, idx = torch.topk(importance, k=keep_dim, largest=True, sorted=True)
    keep_idx = idx.to(device)

    old_W = model.classifier.weight.data  # [C, H]
    W_pruned = old_W[:, keep_idx]        # [C, keep_dim]

    num_classes = old_W.size(0)
    new_classifier = nn.Linear(keep_dim, num_classes, bias=False).to(device)
    new_classifier.weight.data.copy_(W_pruned)

    model.classifier = new_classifier
    model.keep_idx = keep_idx    # forward check

    return keep_idx
