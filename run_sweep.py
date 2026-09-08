import re
import subprocess
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/config.yaml")

# ---- Cấu hình sweep ----
TRAIN_SIZES = [10000, 40000, 100000, 200000, 340000, 680000]
SWEEP_EPOCHS = 3   # giảm so với 5 mặc định để chạy nhanh hơn, chỉ cần thấy xu hướng BLEU
 
 
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
 
 
sweep_results = []
 
for size in TRAIN_SIZES:
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
 
    # 4. Gom kết quả
    metrics = parse_metrics_md(Path(results_dir) / "metrics.md")
    sweep_results.append({
        "train_size_samples": size,       # số mẫu (đã nhân đôi, dùng để chọn trong train_dataset)
        "approx_sentence_pairs": size // 2,  # số cặp câu gốc tương ứng, dùng khi báo cáo/vẽ chart
        **metrics,
    })
    print(f"-> {metrics}")
 
print("\n\n===== TỔNG HỢP =====")
for r in sweep_results:
    print(r)
 
output_path = Path("sweep_results.json")
output_path.write_text(
    __import__("json").dumps(sweep_results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"\nĐã lưu kết quả vào {output_path}")