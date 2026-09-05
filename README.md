# Medical Bidirectional Machine Translation (EN ↔ VI)

Fine-tune mô hình `envit5-base` để dịch máy hai chiều **Anh ↔ Việt trong lĩnh vực y khoa**, sử dụng bộ dữ liệu song ngữ chuyên ngành **MedEV**.

## Dataset

Dự án dùng [**MedEV**](https://huggingface.co/datasets/nhuvo/MedEV) — bộ dữ liệu song ngữ Anh-Việt chuyên ngành y khoa, gồm 358.885 cặp câu được thu thập từ tóm tắt bài báo khoa học, MSD Manuals, tóm tắt luận án y khoa và các bản dịch bài báo chuyên ngành.

| Split | Số cặp câu |
|---|---|
| Train | 340.897 |
| Validation | 8.939 |
| Test | 8.960 |

> Vo, N., Nguyen, D. Q., Le, D. D., Piccardi, M., & Buntine, W. (2024). *Improving Vietnamese-English Medical Machine Translation*. LREC-COLING 2024. [arXiv:2403.19161](https://arxiv.org/abs/2403.19161)

**Lưu ý bản quyền**: dữ liệu được thu thập từ các nguồn công khai (tạp chí khoa học, MSD Manuals...) nhưng **không được phép sử dụng cho mục đích thương mại** — chỉ dùng cho nghiên cứu/giáo dục, theo đúng cam kết của nhóm tác giả gốc.

## Model gốc

Fine-tune từ [`VietAI/envit5-base`](https://huggingface.co/VietAI/envit5-base) — biến thể song ngữ Anh-Việt của T5, pretrain trên 80GB text tiếng Anh + 80GB text tiếng Việt.

## Kiến trúc pipeline

```
MedEV (HF Hub)
   │
   ▼
[1] prepare_data.py  ── tải dữ liệu thô → làm sạch (dedup, loại leakage, loại mislabeled)
   │                     → tokenize 2 chiều (en→vi, vi→en) → lưu xuống data/processed/
   ▼
[2] train.py          ── load data đã tokenize → fine-tune envit5-base bằng Seq2SeqTrainer
   │                     → tự resume từ checkpoint nếu bị ngắt (local hoặc Hugging Face Hub)
   ▼
[3] evaluation.py     ── đánh giá model đã fine-tune trên test set (BLEU / METEOR / TER)
                          cho cả 2 chiều dịch, xuất kết quả ra results/
```

`train.py` **không** tự tải/làm sạch/tokenize dữ liệu — bắt buộc phải chạy `prepare_data.py` trước ít nhất 1 lần.

## Cấu trúc thư mục

```
medical-bidirectional-translator/
├── config/
│   └── config.yaml          # toàn bộ đường dẫn + tham số dùng chung
├── src/
│   ├── __init__.py
│   ├── data_utils.py        # tải dữ liệu thô + làm sạch (dedup, leakage, mislabeled)
│   ├── preprocessing.py     # hàm tokenize 2 chiều dịch
│   ├── prepare_data.py      # script chạy 1 lần: raw → clean → tokenize → lưu disk
│   ├── model_utils.py       # load tokenizer/model gốc và model đã fine-tune
│   ├── metrics.py           # tính BLEU / METEOR / TER
│   ├── evaluation.py        # đánh giá model trên test set
│   └── train.py             # fine-tune model bằng Seq2SeqTrainer
├── data/                     # (sinh ra khi chạy, không commit)
│   └── processed/
├── checkpoints/               # (sinh ra khi chạy, không commit)
├── results/                    # (sinh ra khi chạy) metrics.md, sample_translations.md
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Sau khi cài xong, tải resource cho metric METEOR (chỉ cần chạy 1 lần):

```python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

> **Windows**: nếu gặp lỗi `ImportError: T5Tokenizer requires the SentencePiece library`, chạy `pip install sentencepiece`. Nếu build lỗi trên Python quá mới, cân nhắc dùng Python 3.11/3.12.

## Cấu hình (`config/config.yaml`)

| Mục | Ý nghĩa |
|---|---|
| `paths.*` | Đường dẫn tương đối cho dữ liệu, checkpoint, kết quả — chạy script từ thư mục gốc project |
| `model.base_model_repo` | Model gốc dùng để fine-tune (`VietAI/envit5-base`) |
| `sequence_length.max_source_length/max_target_length` | Độ dài chuỗi tối đa khi tokenize (mặc định 256) |
| `dataset.hf_dataset_repo` | Repo HF của MedEV |
| `huggingface.push_to_hub` | `true` để tự đẩy checkpoint/model lên Hugging Face Hub — **nên bật khi train trên Colab/Kaggle** để resume được qua nhiều session |
| `training.train_size` | Số cặp câu train thực tế sử dụng (điền số lớn hơn kích thước dataset để dùng full) |

## Cách chạy

```bash
# 1. Chuẩn bị dữ liệu (chỉ cần chạy 1 lần)
python -m src.prepare_data

# 2. Fine-tune model (chạy lại vẫn tự resume nếu có checkpoint)
python -m src.train

# 3. Đánh giá trên test set
python -m src.evaluation --model checkpoints/final
# hoặc đánh giá thẳng model đã push lên Hub:
python -m src.evaluation --model <hf-username>/medical-bidirectional-machine-translation
```

## Chạy trên Google Colab / Kaggle

- **Colab**: mount Google Drive, đặt `CORPUS_PATH`/`paths.*` trỏ vào thư mục trên Drive để dữ liệu/checkpoint tồn tại lâu dài giữa các session.
- **Kaggle**: bật **Internet** và **GPU** trong Settings; `/kaggle/input` chỉ đọc nên mọi output (`data/processed`, `checkpoints`, `results`) phải nằm dưới `/kaggle/working`. Vì mỗi session Kaggle giới hạn thời gian chạy, nên **bật `huggingface.push_to_hub: true`** để checkpoint được lưu lên Hub, resume tiếp ở session sau bằng cách chạy lại đúng `python -m src.train`.
- Đăng nhập Hugging Face trước khi train (để push checkpoint):
  ```python
  from huggingface_hub import login
  login("<hf_token>")
  ```

## Kết quả

Sau khi chạy `python -m src.evaluation`, kết quả BLEU/METEOR/TER trên test set (cả 2 chiều dịch) được lưu tại `results/metrics.md`, kèm 5 mẫu dịch minh hoạ mỗi chiều tại `results/sample_translations.md`.

Tham khảo kết quả của nhóm tác giả gốc dataset khi fine-tune `envit5-base` trên toàn bộ 340K câu (5 epoch, batch hiệu dụng 128, 4×A100):

| Direction | BLEU | METEOR | TER↓ |
|---|---|---|---|
| En→Vi | 50.10 | 0.720 | 43.43 |
| Vi→En | 40.66 | 0.666 | 54.07 |

*(kết quả của bạn có thể khác do khác batch size/số câu train/phần cứng — xem phần Implementation notes bên dưới)*

## Ghi chú triển khai
- Theo Figure 1 của paper gốc, dùng 10K câu đầu đã đạt phần lớn lợi ích BLEU so với full 340K câu — có thể giảm `training.train_size` để rút ngắn thời gian train nếu tài nguyên GPU hạn chế.

## Citation

Nếu dùng lại dataset MedEV, vui lòng trích dẫn:

```bibtex
@inproceedings{vo2024medev,
  title     = {Improving Vietnamese-English Medical Machine Translation},
  author    = {Vo, Nhu and Nguyen, Dat Quoc and Le, Dung D. and Piccardi, Massimo and Buntine, Wray},
  booktitle = {Proceedings of LREC-COLING 2024},
  year      = {2024}
}
```

