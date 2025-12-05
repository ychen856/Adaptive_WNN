## python:
initialization: <br/>
Adaptive_WNN> python -m src.train <br/>
evaluation: <br/>
Adaptive_WNN> python -m importer <br/>


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

## Recreating Project in Vivado
Note this project is configured for the Arty Z7-20 on Vivado 2025.2.

### 1. Prerequisites and Setup

Ensure you have the following directory structure, as the build script relies on relative paths to locate resources:

* The local repository folder (e.g., `Adaptive_WNN-2.0/`)
    * `hardware/`
        * `ip_repo/` (Contains custom IP source files)
        * `scripts/` (Contains `build.tcl`)

### 2. Create the Vivado Project

Use the Vivado Tcl Shell to execute the build instructions.

1.  Open the **Vivado Tcl Shell** or the **Tcl Console** inside the Vivado GUI.
2.  Navigate to the `scripts` directory within your local repository clone:
    ```bash
    cd <path/to/your/repo>/hardware/scripts
    ```
3.  Source the build script. This command will execute all instructions, set the **IP Repository paths**, and generate the entire Vivado project structure inside a sub-directory called `vivado_project`.
    ```tcl
    source build.tcl
    ```

### 3. Verify the Block Design

After the script completes, the project will open automatically.

* In the **Sources** pane, locate and open the **Block Design** (`WNNAcceleratorBlk.bd`).
* Verify that the custom IP (`wnn_axi_0`) is instantiated and that the design has resolved all addresses and connections without showing critical warnings.

### 4. Generate the Bitstream

Once verified, the design is ready for synthesis and implementation.

1.  In the Vivado GUI, click **Generate Bitstream**.
2.  The resulting `.bit` file will be located in the implementation run directory (e.g., `vivado_project/WNNAccelerator.runs/impl_1/`). You will also need the .hwh file found under `vivado_project/WNNAccelerator.gen/sources_1/bd/WNNAcceleratorBlk/hw_handoff`.

## Running on PYNQ

### 1. Setup and File Transfer

Upload the contents of the `pynq/` folder to your PYNQ board (e.g., via Jupyter interface or Samba). The directory must include:

* `WNNAcceleratorBlk.bit` & `WNNAcceleratorBlk.hwh`: The pre-compiled hardware overlay.
* `luts/`: A directory containing the quantized Look-Up Tables (`.mem` files) generated during training.
* `mnist_fpga_test_data.npz`: Compressed NumPy array containing the test dataset.
* `wnn_runner.cpp`: The C++ source code for the high-performance inference driver.
* `inference.ipynb`: The Jupyter notebook containing the hybrid Python/C++ control script.

**Note:** The provided data and LUTs are generated using a fixed seed for reproducibility. If you train a new model, ensure you upload the newly generated `luts` folder and test data to the board.

### 2. Execution

1.  Open `inference.ipynb` in the PYNQ Jupyter interface.
2.  Execute the cells. The script will automatically:
    * Program the FPGA bitstream.
    * Load weights into the FPGA BRAM.
    * **Compile the C++ driver** (`wnn_runner.cpp`) using `g++`.
    * Pack the test data into contiguous memory (CMA).
    * Hand off execution to the C++ executable for maximum speed.

### 3. Performance

Upon successful execution, the notebook will report classification accuracy and hardware throughput. Current benchmark metrics on Arty Z7-20:

* **Accuracy:** ~95.97%
* **Throughput:** ~26,124 FPS
* **Latency:** ~38 µs per image
* **Power (PL Dynamic):** ~138 mW