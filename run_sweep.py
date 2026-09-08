import json
import re
import subprocess
import yaml
from pathlib import Path
 
CONFIG_PATH = Path("config/config.yaml")
SWEEP_RESULTS_PATH = Path("sweep_results.json")

# ---- Cấu hình sweep ----
TRAIN_SIZES = [200, 1000]
SWEEP_EPOCHS = 20   
 
 
def run(cmd: str):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)
 
 
def parse_metrics_md(path: Path) -> dict:
    """Đọc bảng markdown do evaluation.py sinh ra, trả về dict {direction: {bleu, meteor, ter}}."""
    text = path.read_text(encoding="utf-8")
    rows = re.findall(
        r"\|\s*(en -> vi|vi -> en)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
        text,
    )
    return {
        direction: {"bleu": float(bleu), "meteor": float(meteor), "ter": float(ter)}
        for direction, bleu, meteor, ter in rows
    }
 
 
def load_existing_results() -> dict:
    """Doc ket qua da luu tu lan chay truoc (neu co) -> {train_size: result_dict}."""
    if SWEEP_RESULTS_PATH.exists():
        existing = json.loads(SWEEP_RESULTS_PATH.read_text(encoding="utf-8"))
        return {r["train_size_samples"]: r for r in existing}
    return {}
 
 
def save_results(results_by_size: dict):
    """Ghi lai TOAN BO ket qua hien co ra file, goi sau MOI muc (khong doi den cuoi)."""
    ordered = [results_by_size[size] for size in TRAIN_SIZES if size in results_by_size]
    SWEEP_RESULTS_PATH.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
 
 
results_by_size = load_existing_results()
if results_by_size:
    done = sorted(results_by_size.keys())
    print(f"Tim thay ket qua da co san cho cac train_size: {done} -> se bo qua, chi chay tiep phan con lai.")
 
for size in TRAIN_SIZES:
    if size in results_by_size:
        print(f"\n[BO QUA] train_size={size} da co ket qua tu lan chay truoc.")
        continue
 
    print(f"\n{'=' * 60}\nTRAIN_SIZE = {size}\n{'=' * 60}")
 
    run_tag = f"size_{size}"
    ckpt_dir = f"checkpoints_{run_tag}"
    results_dir = f"results_{run_tag}"
 
    # 1. Sửa config.yaml cho lần chạy này
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["training"]["train_size"] = size
    cfg["training"]["num_train_epochs"] = SWEEP_EPOCHS
    cfg["paths"]["checkpoint_dir"] = ckpt_dir
    cfg["paths"]["final_model_dir"] = f"{ckpt_dir}/final"
    cfg["paths"]["results_dir"] = results_dir
    cfg["huggingface"]["push_to_hub"] = False   # QUAN TRỌNG: tắt để không lẫn checkpoint giữa các lần
    CONFIG_PATH.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
 
    # 2. Train (subprocess mới -> đọc config vừa sửa)
    run("python -m src.train")
 
    # 3. Evaluate
    run(f"python -m src.evaluation --model {ckpt_dir}/final")
 
    # 4. Gom kết quả + LƯU NGAY (không đợi hết cả sweep) -> chịu được timeout giữa chừng
    metrics = parse_metrics_md(Path(results_dir) / "metrics.md")
    results_by_size[size] = {
        "train_size_samples": size,
        "approx_sentence_pairs": size // 2,
        **metrics,
    }
    save_results(results_by_size)
    print(f"-> Đã lưu kết quả cho train_size={size}: {results_by_size[size]}")
 
print("\n\n===== TỔNG HỢP =====")
for size in TRAIN_SIZES:
    if size in results_by_size:
        print(results_by_size[size])
 
missing = [s for s in TRAIN_SIZES if s not in results_by_size]
if missing:
    print(f"\nCòn thiếu (chưa chạy xong): {missing} -> chạy lại `python run_sweep.py`, sẽ tự tiếp tục từ đây.")
else:
    print("\nĐã hoàn tất toàn bộ sweep.")