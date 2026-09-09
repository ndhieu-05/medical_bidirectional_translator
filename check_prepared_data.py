"""
Kiểm tra dữ liệu sau khi chạy `python -m src.prepare_data`, TRƯỚC khi train.
Chạy: python check_prepared_data.py

Không sửa/ghi gì lên data - chỉ đọc và in ra để bạn tự mắt kiểm tra.
"""

from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict

from src import model_utils

CONFIG_PATH = Path("config/config.yaml")
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ------------------------------------------------------------------
# 1. Kiểm tra clean_dataset (trước khi tokenize)
# ------------------------------------------------------------------
section("1. CLEAN DATASET (data/processed/med_dataset_clean)")

clean_path = config["paths"]["clean_dataset_dir"]
if not Path(clean_path).is_dir():
    raise FileNotFoundError(f"Chưa có {clean_path}. Chạy `python -m src.prepare_data` trước.")

clean_ds: DatasetDict = DatasetDict.load_from_disk(clean_path)

for split in ["train", "validation", "test"]:
    if split in clean_ds:
        print(f"  {split}: {len(clean_ds[split])} cặp câu")

# 1a. Kiểm tra trùng lặp trong từng split
for split in ["train", "validation", "test"]:
    if split not in clean_ds:
        continue
    pairs = list(zip(clean_ds[split]["en"], clean_ds[split]["vi"]))
    n_unique = len(set(pairs))
    if n_unique != len(pairs):
        print(f"  [CẢNH BÁO] {split}: có {len(pairs) - n_unique} cặp câu bị trùng lặp!")
    else:
        print(f"  [OK] {split}: không có trùng lặp nội bộ")

# 1b. Kiểm tra leakage giữa train và validation/test (cặp câu xuất hiện ở cả 2 nơi)
if "train" in clean_ds:
    train_pairs = set(zip(clean_ds["train"]["en"], clean_ds["train"]["vi"]))
    for split in ["validation", "test"]:
        if split not in clean_ds:
            continue
        split_pairs = set(zip(clean_ds[split]["en"], clean_ds[split]["vi"]))
        overlap = train_pairs & split_pairs
        if overlap:
            print(f"  [CẢNH BÁO] LEAKAGE: {len(overlap)} cặp câu xuất hiện ở CẢ train VÀ {split}!")
        else:
            print(f"  [OK] Không có leakage giữa train và {split}")

# 1c. In vài mẫu để mắt thường kiểm tra
print("\n  --- 3 mẫu đầu train ---")
for i in range(min(3, len(clean_ds["train"]))):
    print(f"  EN: {clean_ds['train']['en'][i]}")
    print(f"  VI: {clean_ds['train']['vi'][i]}")
    print()


# ------------------------------------------------------------------
# 2. Kiểm tra tokenized_train (sau khi tokenize, prefix, nhân đôi)
# ------------------------------------------------------------------
section("2. TOKENIZED TRAIN (data/processed/tokenized_train)")

tokenizer, _model, _device = model_utils.load_base_tokenizer_and_model()

train_tok_path = config["paths"]["tokenized_train_dir"]
if not Path(train_tok_path).is_dir():
    raise FileNotFoundError(f"Chưa có {train_tok_path}.")

train_tok = Dataset.load_from_disk(train_tok_path)
n_pairs = len(clean_ds["train"])
n_samples = len(train_tok)

print(f"  Số cặp câu gốc (clean): {n_pairs}")
print(f"  Số mẫu sau tokenize (kỳ vọng = 2x): {n_samples}")
if n_samples == 2 * n_pairs:
    print("  [OK] Đúng 2x như thiết kế (mỗi cặp câu sinh 2 mẫu en->vi và vi->en)")
else:
    print("  [CẢNH BÁO] Số mẫu KHÔNG đúng 2x số cặp câu gốc - kiểm tra lại preprocess_function!")

print("\n  --- Decode thử 4 mẫu đầu (phải thấy xen kẽ en->vi rồi vi->en) ---")
for i in range(min(4, n_samples)):
    row = train_tok[i]
    input_text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
    label_ids = [t for t in row["labels"] if t != -100]
    label_text = tokenizer.decode(label_ids, skip_special_tokens=True)

    ok_prefix = input_text.strip().lower().startswith(("en:", "vi:"))
    flag = "[OK]" if ok_prefix else "[CẢNH BÁO: THIẾU PREFIX]"

    print(f"  mẫu {i} {flag}")
    print(f"    INPUT : {input_text}")
    print(f"    LABEL : {label_text}")
    print()

# 2b. Kiểm tra độ dài token (phát hiện truncation bất thường)
lengths_in = [len(x) for x in train_tok["input_ids"][:2000]]
lengths_lb = [len(x) for x in train_tok["labels"][:2000]]
max_src = config["sequence_length"]["max_source_length"]
max_tgt = config["sequence_length"]["max_target_length"]
n_truncated_in = sum(1 for l in lengths_in if l >= max_src)
n_truncated_lb = sum(1 for l in lengths_lb if l >= max_tgt)
print(f"  Độ dài input_ids (mẫu 2000 đầu): min={min(lengths_in)}, max={max(lengths_in)}, "
      f"trung bình={sum(lengths_in)/len(lengths_in):.1f}")
print(f"  Độ dài labels    (mẫu 2000 đầu): min={min(lengths_lb)}, max={max(lengths_lb)}, "
      f"trung bình={sum(lengths_lb)/len(lengths_lb):.1f}")
print(f"  Số mẫu chạm max_source_length ({max_src}) -> có thể bị truncate: {n_truncated_in}")
print(f"  Số mẫu chạm max_target_length ({max_tgt}) -> có thể bị truncate: {n_truncated_lb}")


# ------------------------------------------------------------------
# 3. Kiểm tra tokenized_val
# ------------------------------------------------------------------
section("3. TOKENIZED VALIDATION (data/processed/tokenized_val_en_vi)")

val_tok_path = config["paths"]["tokenized_val_dir"]
if not Path(val_tok_path).is_dir():
    raise FileNotFoundError(f"Chưa có {val_tok_path}.")

val_tok = Dataset.load_from_disk(val_tok_path)
print(f"  Số mẫu validation: {len(val_tok)}")
print("  (Lưu ý: validation hiện chỉ tokenize 1 chiều en->vi, dùng để track BLEU lúc train)")

print("\n  --- Decode thử 2 mẫu đầu ---")
for i in range(min(2, len(val_tok))):
    row = val_tok[i]
    input_text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
    label_ids = [t for t in row["labels"] if t != -100]
    label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
    print(f"  INPUT : {input_text}")
    print(f"  LABEL : {label_text}")
    print()

section("XONG - kiểm tra kỹ các dòng [CẢNH BÁO] ở trên trước khi chạy train")