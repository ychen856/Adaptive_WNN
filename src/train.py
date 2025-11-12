# src/train/train_wnn.py
from pathlib import Path
import json
from dataio.mapping import make_tuple_mapping, audit_mapping
from core.wisard import WiSARD
from prune import *
from src.core.decision import compute_lut_priority_entropy
from src.tools.export import export_profile_bundle
from src.tools.loader import load_profile_bundle
from src.tools.utils import make_per_lut_kcap
from test import *
from src.core.infer import *

from torchvision import transforms

from test.eval import eval_grid_bits_luts

# from core.decision import tune_decision  #  Step 2

CANONICAL_MAPPING = Path("D:/workspace/Adaptive_WNN/models/meta/tuple_mapping.json")

def load_or_create_mapping(bit_len, tiles, num_luts, addr_bits, seed=42, save_path=CANONICAL_MAPPING):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        mapping = json.loads(save_path.read_text())
        # alignment check
        assert len(mapping) == num_luts, "num_luts mismatch with saved mapping"
        return mapping

    mapping = make_tuple_mapping(
        num_luts=num_luts,
        addr_bits=addr_bits,
        bit_len=bit_len,
        tiles=tiles,          #  None or meta["tile_index_ranges"]
        seed=seed
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping


if __name__ == "__main__":
    # load dataset
    print('data/model initialization...')
    input_path = 'D:/workspace/Adaptive_WNN/datasets'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # encoding
    '''
    x_train_bit_vec, meta = encode_batch(x_train, tiles=(4, 4), levels=8)
    x_test_bit_Vec, _ = encode_batch(x_test, tiles=(4, 4), levels=8)'''

    # encode + sobel
    x_train_bit_vec, meta = encode_batch_thermo_plus_sobel(
        x_train, tiles=(4, 4), levels=8, sobel_threshold_ratio=0.2
    )
    x_test_bit_Vec, _ = encode_batch_thermo_plus_sobel(
        x_test, tiles=(4, 4), levels=8, sobel_threshold_ratio=0.2
    )

    random.seed(time.time())
    seed = random.randint(0, 100)

    # tiling
    NUM_LUTS = 512
    ADDR_BITS = 7
    tuple_mapping = load_or_create_mapping(meta["total_bits"], None, NUM_LUTS, ADDR_BITS, seed)

    stats = audit_mapping(tuple_mapping, meta["total_bits"])
    print("[mapping audit]", stats)

    # optional: save mapping and meta
    with open("D:/workspace/Adaptive_WNN/models/meta/model_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            **meta,
            "address_bits": ADDR_BITS,
            "endianness": "little",
            "tuple_mapping_path": "D:/workspace/Adaptive_WNN/models/meta/model_meta.json"
        }, f, indent=2)

    # train WiSARD
    print('train dataset...')
    C = 10
    model = WiSARD(
        num_classes=C,
        num_luts_per_class=NUM_LUTS,
        address_bits=ADDR_BITS,
        tuple_mapping=tuple_mapping,
        value_dtype=np.uint16,
        endianness="little",
    )
    model.fit(x_train_bit_vec, y_train, batch=512)

    # get accuracy baseline
    full_mask = np.ones(model.num_luts_per_class, dtype=np.float32)
    acc_full = eval_masked(model, x_test_bit_Vec, y_test, full_mask, alpha=1.0)
    print("accuracy baseline(full)=", acc_full)


    ##################################################
    # LUT pruning
    ##################################################
    print('LUT pruning...')
    # get heuristic LUP priority (w lut)
    priority_entropy = compute_lut_priority_entropy(model)

    # accuracy under different entropy
    keep_ratios = [1.0, 0.75, 0.5, 0.25]
    w_lut = model.compute_lut_weights(model, x_train_bit_vec[:1000], y_train[:1000], alpha=1.0)

    for r in keep_ratios:
        mask, _ = make_keep_mask(w_lut, r)
        acc_mask = eval_masked(model, x_test_bit_Vec, y_test, mask)
        print(f"[LUT pruning] LUT_keep_ratio={r:.2f} -> acc={acc_mask:.4f}")


    #######################################
    # bit pruning
    #######################################
    # Given: model, tuple_mapping, bit_priority, X_test_bits, y_test

    lut_priority = compute_lut_priority_entropy(model)
    bit_priority = compute_bit_priority_entropy(x_test_bit_Vec, y_test, C)

    # adaptive bit pruning rate
    caps = make_per_lut_kcap(lut_priority, top_ratio=0.2, low_ratio=0.3,
                             top_cap=7, mid_cap=5, low_cap=4)
    # build with soft coverage + (global or per-LUT) k_cap
    prof = build_runtime_profile_per_lut_adaptive3(
        model, tuple_mapping, bit_priority,
        bits_keep_ratio=1, X_bits_val=x_train_bit_vec[:1000],
        coverage_mode="soft", coverage_r=1, coverage_k_threshold=4,
        H_min=1.8, U_min=48, H_target=2.6, dH_min=0.02,
        k_cap=caps  # or k_cap=caps (per-LUT array from make_per_lut_kcap)
    )
    acc = eval_with_profile_varm(prof, x_test_bit_Vec, y_test, mode="log_posterior")
    stat = profile_stats(prof, n_full=7)
    print(f"acc={acc:.4f} comp={stat['compression']:.3f} avg_m={stat['avg_m']:.2f}")


    ##############################################
    # global pruning w adaptive auto sensing pruning rate
    ##############################################
    prof_pass1 = build_runtime_profile_per_lut_adaptive3(
        model, tuple_mapping, bit_priority,
        bits_keep_ratio=1.0, X_bits_val=x_train_bit_vec[:1000],
        coverage_mode="soft", coverage_r=1, coverage_k_threshold=4,
        H_min=1.8, U_min=48, H_target=2.6, dH_min=0.02,
        k_cap=7  # let LUT find its dynamic k_final
    )
    # get each LUT's k
    k_nat = np.array([row["k_final"] for row in prof_pass1["meta"]], dtype=np.int32)

    # lut_priority
    order = np.argsort(-lut_priority)
    L = len(order)
    top = set(order[:int(0.20 * L)])
    low = set(order[-int(0.30 * L):])

    caps = k_nat.copy()
    for i in range(L):
        if i in top:
            caps[i] = min(7, k_nat[i] + 1)  # for important LUT
        elif i in low:
            caps[i] = max(4, k_nat[i] - 1)  # not important LUT
        else:
            caps[i] = np.clip(k_nat[i], 5, 6)

    prof_pass2 = build_runtime_profile_per_lut_adaptive3(
        model, tuple_mapping, bit_priority,
        bits_keep_ratio=1.0, X_bits_val=x_train_bit_vec,
        coverage_mode="soft", coverage_r=1, coverage_k_threshold=4,
        H_min=1.8, U_min=48, H_target=2.6, dH_min=0.02,
        k_cap=caps  # ← dynamic per-LUT upper bound
    )
    acc = eval_with_profile_varm(prof_pass2, x_test_bit_Vec, y_test, mode="log_posterior")
    stats = profile_stats_total(prof_pass2, n_full=7, L_full=len(tuple_mapping))
    print(f"acc={acc:.4f}  L_comp={stats['L_comp']:.3f}  Addr_comp={stats['addr_comp']:.3f}  "
          f"Total={stats['total_comp']:.3f}  avg_m={stats['avg_m']:.2f}")

    keep_ids = select_top_luts_by_priority(lut_priority, keep_ratio=0.5)
    prof_joint = drop_profile_to_luts(prof_pass2, keep_ids)
    acc = eval_with_profile_varm(prof_joint, x_test_bit_Vec, y_test)
    stats = profile_stats_total(prof_joint, n_full=7, L_full=len(tuple_mapping))
    print(f"[JOINT] acc={acc:.4f}  L_comp={stats['L_comp']:.3f}  "
          f"Addr_comp={stats['addr_comp']:.3f}  Total={stats['total_comp']:.3f}  avg_m={stats['avg_m']:.2f}")


    ############################################
    # final version -- adaptive bit + LUT joint budget pruning
    ############################################
    # given：model, tuple_mapping, bit_priority, lut_priority,
    #             X_train_bits, X_test_bits, y_test
    print('global budget pruning...')

    for bk in [1.0, 0.9, 0.8, 0.7, 0.6]:
        for r in [1.0, 0.9, 0.8, 0.7, 0.6]:
            # ex：JOINT-BUDGET
            prof, keep_ids, k_global = build_joint_budget_profile(
                model, tuple_mapping, bit_priority, lut_priority, x_train_bit_vec[:1000],
                luts_keep_ratio=r, addr_budget_ratio=bk, n_full=7,
                H_target=2.6, coverage_k_threshold=4, coverage_r=1,
                bucket_mapper=bucket_mapper_mnist_thermo
            )

            acc = eval_with_profile_varm(prof, x_test_bit_Vec, y_test, mode="log_posterior")
            stats = profile_stats_total(prof, n_full=7, L_full=len(tuple_mapping))
            print(f"[JOINT-BUDGET] acc={acc:.4f}  L_comp={stats['L_comp']:.3f}  "
                f"Addr_comp={stats['addr_comp']:.3f}  Total={stats['total_comp']:.3f}  "
                f"avg_m={stats['avg_m']:.2f}")

    print('export...')
    print(profile_stats_total(prof_joint, n_full=7, L_full=len(tuple_mapping)))
    out_dir = "D:/workspace/Adaptive_WNN/src/exports/test_export/"


    # === export ===
    # given profile（bit-only or jointly pruned profile）
    export_profile_bundle(
        out_dir=out_dir,
        profile=prof_joint,
        tuple_mapping_pruned=tuple_mapping,  # can be None
        keep_ids=keep_ids,  # if LUT prune, else None
        include_coe=False
    )


    # === import + eval ===
    bundle = load_profile_bundle(out_dir)
    acc = eval_with_profile_varm(bundle, x_test_bit_Vec, y_test)

    print("eval acc =", acc)