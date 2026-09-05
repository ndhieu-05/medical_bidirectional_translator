"""
Đánh giá model đã fine-tune trên tập test, cả 2 chiều en->vi và vi->en.
Dùng lại tập test đã làm sạch ở data/processed/ (từ prepare_data.py), không tải/làm sạch lại.

Cách chạy:
    python -m src.evaluate --model ndhieu1101/medical-bidirectional-machine-translation
    # hoặc trỏ tới checkpoint local:
    python -m src.evaluation --model checkpoints/final
"""

import argparse
import json
import os
from pathlib import Path
import yaml

import torch
from datasets import DatasetDict
from tqdm.auto import tqdm

from . import metrics, model_utils

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH,"r", encoding = "utf-8") as f:
    config = yaml.safe_load(f)

def evaluate_direction(model, tokenizer, device, dataset, src_col, tgt_col, prefix, batch_size = 32, max_length = 256):
    bleu, meteor, ter = metrics.load_metrics()
    preds, refs = [], []
    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc = f"Eval {prefix}"):
        batch = dataset[i: i + batch_size]
        inputs_text = [prefix + s for s in batch[src_col]]

        inputs = tokenizer(
            inputs_text, return_tensors = "pt", padding = True, truncation = True, max_length = max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length = max_length, num_beams = 4, early_stopping = True)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        preds.extend(decoded)
        refs.extend(batch[tgt_col])

    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    meteor_score = meteor.compute(predictions=preds, references=refs)
    ter_score = ter.compute(predictions=preds, references=[[r] for r in refs])

    result = {"bleu": bleu_score["score"], "meteor": meteor_score["meteor"], "ter": ter_score["score"]}
    print(f"\n===== {prefix.strip(': ').upper()} =====")
    print(f"BLEU  : {result['bleu']:.2f}")
    print(f"METEOR: {result['meteor']:.4f}")
    print(f"TER   : {result['ter']:.2f}")

    return preds, refs, result

def _write_results_md(en_vi_metrics, vi_en_metrics, en_vi_samples, vi_en_samples):
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)

    metrics_path = os.path.join(config["paths"]["results_dir"], "metrics.md")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("# Kết quả đánh giá trên test set\n\n")
        f.write("| Direction | BLEU | METEOR | TER |\n")
        f.write("|---|---|---|---|\n")
        f.write(
            f"| en -> vi | {en_vi_metrics['bleu']:.2f} | {en_vi_metrics['meteor']:.4f} | {en_vi_metrics['ter']:.2f} |\n"
        )
        f.write(
            f"| vi -> en | {vi_en_metrics['bleu']:.2f} | {vi_en_metrics['meteor']:.4f} | {vi_en_metrics['ter']:.2f} |\n"
        )
    print(f"-> Đã lưu {metrics_path}")

    samples_path = os.path.join(config["paths"]["results_dir"], "sample_translations.md")
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write("# Ví dụ dịch (5 mẫu đầu mỗi chiều)\n\n")
        f.write("## en -> vi\n\n")
        for src, ref, pred in list(zip(en_vi_samples["src"], en_vi_samples["ref"], en_vi_samples["pred"]))[:5]:
            f.write(f"- **EN**: {src}\n  **GT**: {ref}\n  **PRED**: {pred}\n\n")
        f.write("## vi -> en\n\n")
        for src, ref, pred in list(zip(vi_en_samples["src"], vi_en_samples["ref"], vi_en_samples["pred"]))[:5]:
            f.write(f"- **VI**: {src}\n  **GT**: {ref}\n  **PRED**: {pred}\n\n")
    print(f"-> Đã lưu {samples_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config["huggingface"]["final_model_repo"], help="HF Hub model id hoặc path local")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if not os.path.isdir(config["paths"]["clean_dataset_dir"]):
        raise FileNotFoundError(
            f"Không tìm thấy {config["paths"]["clean_dataset_dir"]}. Hãy chạy `python -m src.prepare_data` trước."
        )

    print(f"== Load model: {args.model} ==")
    tokenizer, model, device = model_utils.load_finetuned_tokenizer_and_model(args.model)

    print("== Load test set đã làm sạch (không tải/clean lại) ==")
    med_dataset: DatasetDict = DatasetDict.load_from_disk(config["paths"]["clean_dataset_dir"])
    test_set = med_dataset["test"]

    en_vi_preds, en_vi_refs, en_vi_metrics = evaluate_direction(
        model, tokenizer, device, test_set, "en", "vi", "en-vi: ", batch_size=args.batch_size
    )
    vi_en_preds, vi_en_refs, vi_en_metrics = evaluate_direction(
        model, tokenizer, device, test_set, "vi", "en", "vi-en: ", batch_size=args.batch_size
    )

    _write_results_md(
        en_vi_metrics,
        vi_en_metrics,
        {"src": test_set["en"], "ref": en_vi_refs, "pred": en_vi_preds},
        {"src": test_set["vi"], "ref": vi_en_refs, "pred": vi_en_preds},
    )

    summary = {"en_vi": en_vi_metrics, "vi_en": vi_en_metrics}
    print("\n== Tổng kết ==")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()


