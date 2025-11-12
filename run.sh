python -m src.tools.fold \
  --in models/weights/ram_tables.bin \
  --meta models/meta/model_meta.json \
  --to_bits 8 \
  --out models/weights/ram_tables_M8.bin