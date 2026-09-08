"""
Script CHẠY MỘT LẦN DUY NHẤT: tải dữ liệu MedEV -> làm sạch -> tokenize -> lưu xuống data/processed/.

train.py sẽ KHÔNG bao giờ tự gọi lại các bước này — nó chỉ load kết quả đã lưu ở đây.
Muốn tokenize lại (vd đổi max_source_length) thì chạy lại chính script này.

Cách chạy:
    python -m src.prepare_data
"""

import functools
import os

from .import data_utils, model_utils, preprocessing
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding = "utf-8") as f:
    config = yaml.safe_load(f)

def main():
    os.makedirs(config["paths"]["processed_dir"], exist_ok = True)

    print("== [1/4] Tải dữ liệu thô từ MedEV ==")
    raw_dataset = data_utils.load_raw_dataset()

    print("== [2/4] Làm sạch dữ liệu (dedup, loại leakage, loại mislabeled) ==")
    clean_dataset = data_utils.build_clean_dataset(raw_dataset)
    clean_dataset.save_to_disk(config["paths"]["clean_dataset_dir"])
    print(f"-> Đã lưu Dataset đã làm sạch tại: {config["paths"]["clean_dataset_dir"]}")
    print(clean_dataset)

    print("== [3/4] Tải tokenizer (dùng model gốc để tokenize) ==")
    tokenizer, _model, _device = model_utils.load_base_tokenizer_and_model()

    print("== [4/4] Tokenize và lưu xuống disk ==")
    bidir_fn = functools.partial(
        preprocessing.preprocess_function,
        tokenizer = tokenizer,
        max_source_length=config["sequence_length"]["max_source_length"],
        max_target_length=config["sequence_length"]["max_target_length"]
        )
    tokenized_train = clean_dataset["train"].map(
        bidir_fn, batched = True, remove_columns = ["en", "vi"]
    )
    tokenized_train.save_to_disk(config["paths"]["tokenized_train_dir"])
    print(f"-> Đã lưu tokenized train tại: {config["paths"]["tokenized_train_dir"]} ({len(tokenized_train)} mẫu)")

    val_fn = functools.partial(
        preprocessing.preprocess_direction,
        src_col="en",
        tgt_col="vi",
        prefix="en: ",
        tokenizer=tokenizer,
        max_source_length=config["sequence_length"]["max_source_length"],
        max_target_length=config["sequence_length"]["max_target_length"]
    )
    tokenized_val = clean_dataset["validation"].map(
        val_fn, batched = True, remove_columns = ["en", "vi"]
    )
    tokenized_val.save_to_disk(config["paths"]["tokenized_val_dir"])
    print(f"-> Đã lưu tokenized validation tại: {config["paths"]["tokenized_val_dir"]} ({len(tokenized_val)} mẫu)")

    print("\nHoàn tất. Từ giờ chạy `python -m src.train` sẽ load thẳng data đã lưu, không tải/tokenize lại.")
    
if __name__ == "__main__":
    main()
