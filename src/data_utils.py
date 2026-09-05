"""
Load dữ liệu thô từ MedEV + các hàm làm sạch dữ liệu.
Các hàm dùng để Investigate Dataset
"""

import re
import unicodedata
import pandas as pd

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
"""
Pipeline tạo đường dẫn CONFIG_PATH:
file → đường dẫn file Python hiện tại
→ Path(file) → chuyển thành Path object (src/train.py)
→ .resolve() → chuyển thành đường dẫn tuyệt đối (/home/user/project/src/train.py)
→ .parent → đi lên 1 cấp, lấy thư mục chứa file Python (/home/user/project/src)
→ .parent → đi lên thêm 1 cấp, lấy thư mục gốc của project (/home/user/project)
→ / "config" → truy cập thư mục config (/home/user/project/config)
→ / "config.yaml" → trỏ tới file config.yaml (/home/user/project/config/config.yaml)

Kết quả: CONFIG_PATH là Path object trỏ tới file config.yaml.
"""

with open(CONFIG_PATH, "r", encoding = "utf-8") as f:
    config = yaml.safe_load(f)

VI_PATTERN = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữự"
    r"ỳýỷỹỵđ]",
    re.IGNORECASE,
)

def load_raw_dataset() -> DatasetDict:
    """Tải MedEV (en + vi) từ HF Hub và ghép thành 1 DatasetDict với cột en/vi."""
    en_data = load_dataset(
        config["dataset"]["hf_dataset_repo"],
        data_files = config["dataset"]["data_files_en"]
    )

    vi_data = load_dataset(
        config["dataset"]["hf_dataset_repo"],
        data_files = config["dataset"]["data_files_vi"]
    )

    en_data = en_data.rename_column("text", "en")
    vi_data = vi_data.rename_column("text", "vi")

    for split in ["train", "validation", "test"]:
        assert len(en_data[split]) == len(vi_data[split]), (
            f"{split} bị lệch số dòng: en={len(en_data[split])}, vi={len(vi_data[split])}"
        )

    med_dataset = DatasetDict(
            {
                split: concatenate_datasets([en_data[split], vi_data[split]], axis=1)
                for split in ["train", "validation", "test"]
            }
        )
    return med_dataset

def normalize_text(x) -> str:
    """
    Nhận một dữ liệu bất kỳ → chuyển thành chuỗi → chuẩn hóa Unicode → gom các khoảng trắng thừa → xóa khoảng trắng đầu/cuối → trả về chuỗi sạch.
    """
    x = unicodedata.normalize("NFC", str(x)) # Đảm bảo x là chuỗi, sau đó chuẩn hóa Unicode của nó về NFC và lưu kết quả lại vào x  
    x = re.sub(r"\s+", " ", x).strip()
    return x

def clean_df(df: pd.DataFrame, remove_identical_vi: bool = True) -> pd.DataFrame:
    """
    Làm sạch DataFrame song ngữ để giảm dữ liệu trùng lặp và loại bỏ một số cặp dữ liệu có khả năng bị lỗi trước khi train model dịch máy.
    """
    df = df.copy()
    
    df["en"] = df["en"].map(normalize_text)
    df["vi"] = df["vi"].map(normalize_text)

    df = df.drop_duplicates(subset=["en", "vi"]).reset_index(drop=True)

    if remove_identical_vi:
        mask = (df["en"] == df["vi"]) & (df["en"].str.contains(VI_PATTERN, na=False))
        print("Remove suspicious identical Vietnamese rows:", mask.sum())
        df = df[~mask].reset_index(drop=True)

    return df

def _to_df(dataset: DatasetDict, split: str)-> DatasetDict:
    return pd.DataFrame(
        {
            "en": dataset[split]["en"],
            "vi": dataset[split]["vi"]
        }
    )

def build_clean_dataset(med_dataset: DatasetDict)->DatasetDict:
    """
    Thực hiện toàn bộ pipeline làm sạch:
    normalize -> drop duplicate -> loại leakage giữa các split -> loại dòng mislabeled.
    """
    dfs = {
        s: _to_df(med_dataset, s) for s in ["train", "validation", "test"]
    }

    train_df = clean_df(dfs["train"])
    val_df = clean_df(dfs["validation"])
    test_df = clean_df(dfs["test"])

    # Loại leakage: cặp câu train trùng với val/test
    train_df["pair_key"] = train_df["en"] + "|||" + train_df["vi"] # Tạo key cho mỗi dòng của tập Train
    val_keys = set(val_df["en"] + "|||" + val_df["vi"]) # Tạo các cặp câu cho validation
    test_keys = set(test_df["en"] + "|||" + test_df["vi"]) #  Tạo các cặp câu cho test
    leak_keys = val_keys | test_keys # Gộp validation vs test 

    n_leak = train_df["pair_key"].isin(leak_keys).sum() # Kiểm tra train có bị leakage không ? và đếm số dòng bị leak
    print("Remove train leakage:", n_leak)
    train_df = train_df[~train_df["pair_key"].isin(leak_keys)].drop(columns=["pair_key"]).reset_index(drop=True)

    # Loại các dòng mislabeled: en == vi nhưng thực chất lại là tiếng Việt
    mislabeled_mask = (
        train_df["en"].str.strip() == train_df["vi"].str.strip()
        ) & (
        train_df["en"].str.contains(VI_PATTERN, na=False)
    )
    n_mislabeled = mislabeled_mask.sum()
    print("Remove mislabeled:", n_mislabeled)
    train_df = train_df[~mislabeled_mask].reset_index(drop=True)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )
