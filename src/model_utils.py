"""
Load model + tokenizer EnviT5.

Lưu ý: tokenizer.json của VietAI/envit5-base được export từ bản `tokenizers` cũ,
không tương thích với parser Unigram của bản mới -> gây KeyError: 0 khi dùng
AutoTokenizer.from_pretrained() trực tiếp. Cách xử lý: xoá tokenizer.json trong
bản snapshot local, ép dùng T5Tokenizer (chậm hơn FastTokenizer một chút nhưng
không ảnh hưởng chất lượng model).
"""

import os

import torch
from huggingface_hub import snapshot_download
from transformers import  AutoModelForSeq2SeqLM, T5Tokenizer

from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding = "utf-8") as f:
    config = yaml.safe_load(f)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_base_tokenizer_and_model(model_repo: str = config["model"]["base_model_repo"], device: str | None = None):
    """
    Tải model gốc từ HF Hub (dùng khi train từ đầu / prepare_data.py cần tokenizer).
    """
    device = device or get_device()

    path = snapshot_download(model_repo, ignore_patterns=["tokenizer.json"])

    tokenizer = T5Tokenizer.from_pretrained(path, use_fast = False, legacy = False)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)

    return tokenizer, model, device

def load_finetuned_tokenizer_and_model(model_name: str, device: str  | None = None):
    """
    Tải model đã fine tune từ HF Hub hoặc từ path local
    """
    from transformers import AutoTokenizer

    device = device or get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device