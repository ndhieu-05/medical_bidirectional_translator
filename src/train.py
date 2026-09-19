"""
Script train, chạy độc lập.

QUAN TRỌNG: script này KHÔNG tự tải/làm sạch/tokenize dữ liệu.
Nó chỉ load thẳng data đã được `prepare_data.py` lưu sẵn ở data/processed/.
Nếu chưa chạy prepare_data.py, script sẽ dừng lại và báo lỗi rõ ràng thay vì
âm thầm tokenize lại (tốn thời gian + không nhất quán giữa các lần train).

Cách chạy:
    python -m src.prepare_data      # chỉ cần chạy 1 lần
    python -m src.train             # chạy nhiều lần tuỳ ý, không tốn lại bước preprocess
"""

import os

from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from . import metrics, model_utils
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding = "utf-8") as f:
    config = yaml.safe_load(f)

def load_preprocessed_data():
    """ 
    Load dữ liệu train và validation đã được tokenize từ ổ đĩa. 
    
    Kiểm tra xem các thư mục chứa dữ liệu đã tokenize có tồn tại hay không. 
    Nếu chưa có, báo lỗi và hướng dẫn chạy bước chuẩn bị dữ liệu trước. 
    
    Returns: 
        tuple: 
            - train_dataset: Dataset train đã được tokenize. 
            - val_dataset: Dataset validation đã được tokenize. 
    
    Raises:
        FileNotFoundError: Nếu một trong các thư mục dữ liệu đã tokenize không tồn tại. 
    """
    missing = [
        p
        for p in [config["paths"]["tokenized_train_dir"], config["paths"]["tokenized_val_dir"]]
        if not os.path.isdir(p)
    ]
    if missing:
        raise FileNotFoundError(
            "Chưa tìm thấy data đã tokenize tại: "
            + ", ".join(missing)
            + "\nHãy chạy `python -m src.prepare_data` trước (chỉ cần chạy 1 lần)."
        )

    train_dataset = Dataset.load_from_disk(config["paths"]["tokenized_train_dir"])
    val_dataset = Dataset.load_from_disk(config["paths"]["tokenized_val_dir"])
    return train_dataset, val_dataset

def build_training_args() -> Seq2SeqTrainingArguments:
    """ 
    Tạo và trả về cấu hình huấn luyện cho mô hình Seq2Seq. 
    Cấu hình bao gồm các thiết lập về: 
    - Thư mục lưu checkpoint. 
    - Tần suất đánh giá và lưu checkpoint. 
    - Learning rate và learning-rate scheduler. 
    - Batch size và gradient accumulation. 
    - Label smoothing và weight decay. 
    - Số epoch huấn luyện. 
    - Cấu hình sinh văn bản khi đánh giá. 
    - Mixed precision (FP16). 
    - DataLoader và tối ưu việc gom batch theo độ dài. 
    - Cách lựa chọn và lưu lại checkpoint tốt nhất. 
    - Logging và tích hợp Hugging Face Hub. 
    
    Returns: Seq2SeqTrainingArguments: Đối tượng chứa toàn bộ tham số dùng để cấu hình quá trình huấn luyện Seq2Seq. 
    """
    return Seq2SeqTrainingArguments(
        output_dir=config["paths"]["checkpoint_dir"],
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2,
        gradient_checkpointing=False,
        label_smoothing_factor=0.1,
        weight_decay=0.01,
        num_train_epochs=config["training"].get("num_train_epochs", 5),
        generation_max_length=config["sequence_length"]["max_target_length"],
        generation_num_beams=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        predict_with_generate=True,
        fp16=True,
        dataloader_num_workers=4,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=100,
        disable_tqdm=True,
        report_to="none",
        push_to_hub=config["huggingface"]["push_to_hub"],
        hub_model_id=config["huggingface"]["checkpoint_repo"] if config["huggingface"]["push_to_hub"] else None,
        hub_strategy="all_checkpoints",
        hub_always_push=config["huggingface"]["push_to_hub"],
    )

def main():
    print("== Load data đã preprocess từ data/processed/ (không tokenize lại) ==")
    train_dataset, val_dataset = load_preprocessed_data()

    n = min(config["training"]["train_size"], len(train_dataset))
    train_dataset = train_dataset.select(range(n))
    print(f"-> train: {len(train_dataset)} mẫu | validation: {len(val_dataset)} mẫu")

    print("== Load model + tokenizer ==")
    tokenizer, model, device = model_utils.load_base_tokenizer_and_model()
    print(f"-> Đang dùng thiết bị: {device}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    compute_metrics = metrics.build_compute_metrics(tokenizer)
    training_args = build_training_args()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    # Tự resume nếu có checkpoint sẵn (local hoặc trên Hub, nếu PUSH_TO_HUB=true)
    last_checkpoint = None
    if config["huggingface"]["push_to_hub"]:
        try:
            from huggingface_hub import HfApi
            import re as _re

            files = HfApi().list_repo_files(config["huggingface"]["checkpoint_repo"])
            folders = {f.split("/")[0] for f in files if "/" in f}

            if "last-checkpoint" in folders:
                target_folder = "last-checkpoint"
            else:
                ckpt_folders = [f for f in folders if _re.fullmatch(r"checkpoint-\d+", f)]
                target_folder = max(ckpt_folders, key=lambda x: int(x.split("-")[1])) if ckpt_folders else None

            if target_folder:
                snapshot_download(
                    repo_id=config["huggingface"]["checkpoint_repo"],
                    local_dir=config["paths"]["checkpoint_dir"],
                    local_dir_use_symlinks=False,
                    allow_patterns=[f"{target_folder}/*"],   # CHI tai dung 1 checkpoint can, khong tai het lich su
                )
                print(f"-> Đã tải checkpoint để resume: {target_folder}")
            else:
                print("-> Repo trên Hub chưa có checkpoint nào, sẽ train từ đầu.")
        except Exception as e:
            print("-> Chưa có checkpoint trên Hub hoặc lỗi kết nối:", e)

    last_ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], "last-checkpoint")
    if os.path.isdir(last_ckpt_path):
        last_checkpoint = last_ckpt_path
    elif os.path.isdir(config["paths"]["checkpoint_dir"]):
        last_checkpoint = get_last_checkpoint(config["paths"]["checkpoint_dir"])

    if last_checkpoint:
        print(f"Tiếp tục huấn luyện từ: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Không có checkpoint hợp lệ. Train từ đầu.")
        trainer.train()

    print(f"== Lưu model cuối cùng vào {config["paths"]["final_model_dir"]} ==")
    trainer.save_model(config["paths"]["final_model_dir"])
    tokenizer.save_pretrained(config["paths"]["final_model_dir"])

    if config["huggingface"]["push_to_hub"]:
        model.push_to_hub(config["huggingface"]["final_model_repo"])
        tokenizer.push_to_hub(config["huggingface"]["final_model_repo"])

if __name__ == "__main__":
    main()
