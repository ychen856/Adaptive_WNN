## python:
Adaptive_WNN>python -m src.train <br/>
export files Adaptive_WNN/src/exports/foldername

## FPGA usage:
### addr_bits_per_lut.json 
Array of integers m_l (length = num_luts). Each m_l is the address width (bits) for LUT l.

### kept_bits.json 
List[List[int]], length = num_luts.
For LUT l, kept_bits[l] is a global bit index list (length = m_l) describing which input bit positions form the address. Order is LSB-first (i.e., kept_bits[l][0] = bit used for address bit a0).

### keep_ids.json (optional)
Indices (w.r.t. original L_full) that were kept after LUT pruning.
Useful if the FPGA pipeline wants to preserve original LUT numbering or for debugging.

### tuple_mapping_pruned.json (optional)
If you exported the post-prune tuple mapping, it is a List[List[int]] of length num_luts. This is mostly for diagnostics; FPGA should rely on kept_bits.json, not the original tuple mapping.

### luts/lut_XXX.npy
numpy arrays for each LUT l with shape (C, 2^m_l) and dtype float32 (or int32 depending on export).
Entry [c, addr] is the class count (or smoothed count) for class c at address addr.

### coe/lut_XXX.coe (optional)
Memory initialization files for FPGA BRAMs. Each .coe lists rows for addresses 0..(2^m_l-1), and each row contains C comma-separated integers (counts for all classes at that address), radix per file header.
You may choose a different packing (e.g., one BRAM per class vs one wide BRAM containing all C). The provided .coe is a neutral, human-legible format—feel free to adapt packing to your synthesis flow.
Need to be converted to the .mem files for FPGA usage.


