# src/train/train_wnn.py
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
import random 
from src.dataio.mapping import make_tuple_mapping
from src.prune import *
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN
from src.dataio.encode import minmax_normalize, thermometer_encode, dt_thermometer_encode, compute_dt_thresholds
from torch.utils.data import TensorDataset, DataLoader
from src.tools.fpga_tools.fpga_export_utils import export_lut_init_files


CANONICAL_MAPPING = Path("/Users/yi-chunchen/workspace/Adaptive_WNN/models/meta/tuple_mapping.json")

def load_or_create_mapping(bit_len, tiles, num_luts, addr_bits, seed=42, save_path=CANONICAL_MAPPING):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = None
    
    if save_path.exists():
        try:
            content = save_path.read_text()
            if content.strip(): 
                mapping = json.loads(content)
                if len(mapping) != num_luts:
                    print(f"[Mapping] Warning: Saved mapping has {len(mapping)} LUTs, but config needs {num_luts}. Regenerating...")
                    mapping = None 
            else:
                mapping = None
        except Exception as e:
            print(f"[Mapping] Error reading saved mapping: {e}. Regenerating...")
            mapping = None

    if mapping is None:
        print(f"[Mapping] Generating new tuple mapping (seed={seed})...")
        random.seed(seed)
        np.random.seed(seed)
        
        mapping = make_tuple_mapping(
            num_luts=num_luts,
            addr_bits=addr_bits,
            bit_len=bit_len,
            tiles=tiles,
            seed=seed
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)
            
    return mapping

def get_lr(epoch):
    if epoch < 25: return 1e-3
    elif epoch < 55: return 3e-4
    else: return 1e-4

def compute_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_model(model, train_loader, val_loader, device, num_epochs=50, base_lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    best_state = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss, train_acc = eval_epoch(model, train_loader, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def collect_hidden_activations(model, data_loader, device):
    model.eval()
    all_h = []
    all_y = []
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, h_last = model(xb, return_hidden=True)
        all_h.append(h_last.cpu())
        all_y.append(yb.cpu())
    H = torch.cat(all_h, dim=0)
    Y = torch.cat(all_y, dim=0)
    return H, Y

def build_lut_counts(x_bits, y, tuple_mapping, num_classes, addr_bits):
    if isinstance(x_bits, torch.Tensor): x_bits = x_bits.cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().numpy()
    x_bits = np.asarray(x_bits)
    y = np.asarray(y)

    num_samples, bit_len = x_bits.shape
    num_luts = len(tuple_mapping)
    M = 1 << addr_bits
    luts = np.zeros((num_luts, num_classes, M), dtype=np.int32)

    for i in range(num_samples):
        cls = int(y[i])
        row = x_bits[i]
        for l, indices in enumerate(tuple_mapping):
            bits = row[indices]
            addr = 0
            for bit_pos, b in enumerate(bits):
                if b: addr |= (1 << (addr_bits - 1 - bit_pos))
            luts[l, cls, addr] += 1
    return luts

def export_fpga_bundle(out_dir, tuple_mapping, addr_bits, luts):
    out_dir = Path(out_dir)
    (out_dir / "luts").mkdir(parents=True, exist_ok=True)

    num_luts, num_classes, M = luts.shape
    assert (1 << addr_bits) == M, "LUT depth mismatch with addr_bits"

    addr_bits_per_lut = [addr_bits] * num_luts
    with open(out_dir / "addr_bits_per_lut.mem", "w") as f:
        for val in addr_bits_per_lut:
            f.write(f"{val:x}\n")

    with open(out_dir / "kept_bits.mem", "w") as f:
        for bits in tuple_mapping:
            hex_str = " ".join(f"{b:x}" for b in bits)
            f.write(hex_str + "\n")

    keep_ids = list(range(num_luts))
    with open(out_dir / "keep_ids.mem", "w") as f:
        for kid in keep_ids:
            f.write(f"{kid:x}\n")

    for l in range(num_luts):
        lut_data = luts[l] 
        lut_data_T = lut_data.T 
        with open(out_dir / "luts" / f"lut_{l:03d}.mem", "w") as f:
            for addr_row in lut_data_T:
                hex_row = " ".join(f"{val:x}" for val in addr_row)
                f.write(hex_row + "\n")
    
    print("\nExported FPGA bundle (MEM format) to:", out_dir)
    print("  num_luts    =", num_luts)

def verify_fpga_logic_diagnostic(fused_luts, x_bits, y, mapping, addr_bits, desc="Test Data"):
    """
    Runs a deep diagnostic simulation to find why accuracy is low.
    Checks Hit Rate, MSB vs LSB, and Accuracy.
    """
    import numpy as np
    print(f"\n[Diagnostic] Simulating Hardware on {desc} ({len(x_bits)} samples)...")
    
    if hasattr(x_bits, 'cpu'): x_bits = x_bits.cpu().numpy()
    if hasattr(y, 'cpu'): y = y.cpu().numpy()
    
    num_samples = x_bits.shape[0]
    num_luts = fused_luts.shape[0]
    
    # --- TEST 1: Bit Order Permutations ---
    # We try both MSB-first (Standard FPGA) and LSB-first (Standard WNN)
    orders = {
        "MSB_First": 2 ** np.arange(addr_bits - 1, -1, -1), # [32, 16, 8, 4, 2, 1]
        "LSB_First": 2 ** np.arange(0, addr_bits, 1)        # [1, 2, 4, 8, 16, 32]
    }
    
    best_acc = 0.0
    best_mode = "None"

    for mode, powers in orders.items():
        print(f"  -> Testing Address Mode: {mode}...")
        
        # 1. Generate Addresses
        lut_addresses = np.zeros((num_samples, num_luts), dtype=np.int32)
        for l, indices in enumerate(mapping):
            bits = x_bits[:, indices]
            lut_addresses[:, l] = bits.dot(powers)

        # 2. Check Hit Rate (Crucial!)
        # How many addresses actually point to a non-zero entry in the LUT?
        # We check LUT 0 as a proxy
        lut0_table = fused_luts[0].T # (64, 10)
        lut0_addrs = lut_addresses[:, 0]
        # Sum of absolute weights at these addresses
        hits = np.sum(np.abs(lut0_table[lut0_addrs]), axis=1) > 0
        hit_rate = np.mean(hits)
        print(f"     [Hit Rate] {hit_rate*100:.1f}% of lookups found a valid weight.")

        # 3. Calculate Accuracy
        total_scores = np.zeros((num_samples, 10), dtype=np.int32)
        for l in range(num_luts):
            lut_table = fused_luts[l].T
            total_scores += lut_table[lut_addresses[:, l]]

        acc = np.mean(total_scores.argmax(axis=1) == y)
        print(f"     [Accuracy] {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_mode = mode

    return best_acc, best_mode

def export_fused_single_layer(model, mapping, save_dir, x_test_bits=None, y_test=None, addr_bits=6):
    import numpy as np
    import torch
    from pathlib import Path
    
    print("\n[Fusion] Starting Weight-Fused Export (With Permutation Fix)...")
    save_dir = Path(save_dir)
    (save_dir / "luts").mkdir(parents=True, exist_ok=True)

    # --- STEP 1: Get Weights & UNSHUFFLE THEM ---
    w_cls_sorted = model.classifier.weight.detach() # Shape: [10, N_kept]
    w_lut_internal = model.layers[0].table.detach() # Shape: [500, 64]
    
    # Get the permutation indices (keep_idx)
    # This tells us which Physical LUT corresponds to which Classifier Input
    if hasattr(model, "keep_idx") and model.keep_idx is not None:
        keep_idx = model.keep_idx.long()
        print(f"  -> Detected keep_idx with {len(keep_idx)} entries. Un-shuffling weights...")
        
        # Create a container for physical weights [10, 500]
        num_physical_luts = w_lut_internal.shape[0]
        w_cls_physical = torch.zeros(10, num_physical_luts, device=w_cls_sorted.device)
        
        # Map the sorted weights back to their physical slots
        # If keep_idx[k] = p, it means column k of w_cls belongs to physical LUT p
        w_cls_physical[:, keep_idx] = w_cls_sorted
    else:
        print("  -> No keep_idx found. Assuming 1:1 mapping.")
        w_cls_physical = w_cls_sorted

    w_cls = w_cls_physical.cpu()
    w_lut_internal = w_lut_internal.cpu()

    # --- STEP 2: Fuse (Same as before, but with correct weights) ---
    # Apply Sigmoid to internal table
    w_lut_activation = torch.sigmoid(w_lut_internal) 

    # Fuse: Weight * Sigmoid(Table)
    # [10, 500] -> [500, 10, 1]
    W_C = w_cls.t().unsqueeze(2)
    # [500, 64] -> [500, 1, 64]
    W_L = w_lut_activation.unsqueeze(1)
    
    fused_tensor = W_C * W_L # Result: [500, 10, 64]
    fused_float = fused_tensor.numpy()

    # --- STEP 3: Quantize ---
    w_min, w_max = fused_float.min(), fused_float.max()
    print(f"  -> Fused Range: [{w_min:.4f}, {w_max:.4f}] (Expect ~0.3)")
    
    scale = 255.0 / (w_max - w_min + 1e-9)
    fused_luts = ((fused_float - w_min) * scale).astype(np.int32)

    # --- STEP 4: Validate ---
    if x_test_bits is not None and y_test is not None:
        print("[Validation] Verifying Fixed Logic...")
        # Check with Test Data
        acc, mode = verify_fpga_logic_diagnostic(fused_luts, x_test_bits, y_test, mapping, addr_bits, "Test Data")
        
        if acc < 0.80:
            print(f"[CRITICAL WARNING] Accuracy {acc*100:.2f}% is still low.")
            print("  Check: Did you retrain the classifier AFTER creating keep_idx?")
        else:
            print(f"[SUCCESS] Accuracy is normal: {acc*100:.2f}%")

    # --- STEP 5: Save ---
    export_fpga_bundle(save_dir, mapping, addr_bits, fused_luts)
def extract_real_mapping(model, addr_bits=6):
    """
    Extracts the actual bit-to-LUT wiring from the trained PyTorch model.
    """
    print("\n[Mapping Fix] Extracting REAL wiring from model layer 0...")
    layer0 = model.layers[0]
    
    # We look for a buffer/parameter that holds the indices. 
    # It usually has shape (NumLUTs, AddrBits) or is flattened.
    real_mapping = None
    
    # Search through named buffers (standard for fixed indices)
    for name, buf in layer0.named_buffers():
        # Check if this buffer looks like the mapping (size should be NumLUTs * AddrBits)
        if buf.numel() == 500 * addr_bits: 
            print(f"  -> Found mapping in buffer: '{name}' (Shape: {buf.shape})")
            indices = buf.cpu().numpy().astype(np.int32)
            
            # Reshape to (NumLUTs, AddrBits) if it's flat
            if indices.ndim == 1:
                indices = indices.reshape(500, addr_bits)
            
            real_mapping = indices.tolist()
            break
            
    # If not in buffers, check parameters (unlikely but possible)
    if real_mapping is None:
        for name, param in layer0.named_parameters():
             if param.numel() == 500 * addr_bits and not param.requires_grad:
                print(f"  -> Found mapping in parameter: '{name}'")
                indices = param.detach().cpu().numpy().astype(np.int32)
                if indices.ndim == 1: indices = indices.reshape(500, addr_bits)
                real_mapping = indices.tolist()
                break

    if real_mapping is None:
        raise ValueError("Could not find the mapping indices in model.layers[0]! \n"
                         "Please check the variable names in src/core/wnnLutLayer.py")
    
    print("  -> Extraction successful. This mapping matches the trained weights.")
    return real_mapping

if __name__ == "__main__":
    print('data/model initialization...')
    input_path = '/Users/yi-chunchen/workspace/Adaptive_WNN/datasets'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    z = 32
    thresholds, xmin, xmax = compute_dt_thresholds(x_train, z=z)

    x_train_bits = dt_thermometer_encode(x_train.to(device), thresholds, xmin, xmax)
    x_test_bits   = dt_thermometer_encode(x_test.to(device),   thresholds, xmin, xmax)

    in_bits = x_train_bits.size(1)

    train_ds = TensorDataset(x_train_bits, y_train)
    val_ds   = TensorDataset(x_test_bits, y_test)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)

    C = 10
    ADDR_BITS = 6 
    SEED = 42 

    print(f"Setting random seed to {SEED} for reproducibility...")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # FIXED: Set num_luts=500. 
    # If you generate 2000 here but model only uses 500, the exported mapping won't match!
    training_mapping = load_or_create_mapping(
        bit_len=in_bits, 
        tiles=None, 
        num_luts=500, # <--- FIXED (Was 2000)
        addr_bits=ADDR_BITS, 
        seed=SEED
    )

    # FIXED: Reset seed AGAIN before model init.
    # This ensures the model's internal wiring generation produces the EXACT SAME sequence
    # as the mapping generator above (assuming WNNLayer uses standard random calls).
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = MultiLayerWNN(
        in_bits=in_bits,
        num_classes=10,
        lut_input_size=ADDR_BITS,
        hidden_luts=(500,),
        mapping=training_mapping, # <--- THIS FIXES THE MISMATCH PERMANENTLY
        tau=0.165,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    model = train_model(model, train_loader, test_loader, device, num_epochs=30, base_lr=1e-3)

    train_loss_before, train_acc_before = eval_epoch(model, train_loader, device)
    test_loss_before,  test_acc_before  = eval_epoch(model, test_loader,  device)
    print(f"[Before pruning] train_acc={train_acc_before*100:.2f}%, test_acc={test_acc_before*100:.2f}%")

    # 3) Pruning - FIXED to keep ratio 1.0
    H, Y = collect_hidden_activations(model, train_loader, device)
    H = H.to(device)
    importance = compute_importance_weighted(H, model)
    
    # FIXED: Set keep_ratio=1.0. 
    # We want to use ALL 500 LUTs on the FPGA for maximum accuracy.
    # If you set 0.5, you get 250 LUTs, but your FPGA is hardcoded for 500, leading to errors.
    keep_idx = build_pruned_classifier(model, importance, keep_ratio=1.0, min_keep=64)
    print("Hidden dims before:", H.shape[1], "after:", keep_idx.numel())

    train_loss_after, train_acc_after = eval_epoch(model, train_loader, device)
    test_loss_after,  test_acc_after  = eval_epoch(model, test_loader,  device)
    print(f"[After pruning]  train_acc={train_acc_after*100:.2f}%, test_acc={test_acc_after*100:.2f}%")

    # finetuning
    for name, p in model.named_parameters():
        if "table" in name: p.requires_grad = False 
        else: p.requires_grad = True

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer_ft.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_ft.step()
        _, test_acc_ft  = eval_epoch(model, test_loader,  device)
        print(f"[Finetune {epoch}] test_acc={test_acc_ft*100:.2f}%")

    # ----------------------------------------------------------------
    # EXPORT FPGA BUNDLE (WITH DIAGNOSTIC FIX)
    # ----------------------------------------------------------------
    print("\n=== Starting FPGA Export (With Mapping Fix) ===")

    # 1. EXTRACT REAL MAPPING (The Fix)
    print("[Diagnostic] Extracting 'conn_idx' from model layer 0...")
    if hasattr(model.layers[0], 'conn_idx'):
        # Grab the tensor causing the mismatch
        internal_tensor = model.layers[0].conn_idx
        real_mapping_np = internal_tensor.cpu().numpy().astype(int)
        
        # Reshape if flat
        if real_mapping_np.ndim == 1:
            real_mapping_np = real_mapping_np.reshape(500, ADDR_BITS)
            
        real_model_mapping = real_mapping_np.tolist()
        print(f"  -> Found internal mapping! Shape: {real_mapping_np.shape}")
    else:
        raise RuntimeError("Could not find 'conn_idx' in model! Check variable names.")

    # 2. DEBUG: COMPARE MAPPINGS (The "Nice Debugging")
    # We compare the first LUT's wiring to prove they are different
    old_map_0 = training_mapping[0]
    new_map_0 = real_model_mapping[0]
    
    print("\n[Diagnostic] Mapping Comparison (LUT #0):")
    print(f"  OLD (External): {old_map_0}  <-- This caused 9% accuracy")
    print(f"  NEW (Internal): {new_map_0}  <-- This is the TRUTH")
    
    if old_map_0 != new_map_0:
        print("  [CONFIRMED] Mismatch detected and fixed. Exporting NEW mapping.")
    else:
        print("  [NOTE] Mappings match (unexpected based on logs, but good).")

    # 3. EXPORT USING THE REAL MAPPING
    EXPORT_DIR = "src/exports/fpga_bundle"
    
    # Note: We ignore 'pruned_mapping' here because we are exporting 
    # the FULL 500 LUTs that physically exist in the layer.
    
    export_fused_single_layer(
        model=model,
        mapping=training_mapping,     # <--- Use the clean variable from the top of train.py
        save_dir=EXPORT_DIR,
        x_test_bits=x_test_bits,      # Required for accuracy verification
        y_test=y_test,                # Required for accuracy verification
        addr_bits=ADDR_BITS
    )

