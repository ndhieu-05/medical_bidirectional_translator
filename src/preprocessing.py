"""
Hàm tokenize dữ liệu song phương (en->vi và vi->en).
Nhận `tokenizer` làm tham số thay vì biến global, để tái sử dụng được ở nhiều nơi
(prepare_data.py lúc tokenize 1 lần, evaluate.py lúc cần tokenize lại theo hướng khác).
"""

def preprocess_function(batch, tokenizer, max_source_length, max_target_length):
    """
    Sinh 2 mẫu training cho mỗi cặp câu: 1 mẫu en->vi và 1 mẫu vi->en
    Dùng cho tập train
    """
    inputs = []
    targets = []

    for en, vi in zip(batch["en"], batch["vi"]):
        inputs.append("en: " + en)
        targets.append(vi)

        inputs.append("vi: " + vi)
        targets.append(en)

    model_inputs =  tokenizer(inputs, max_length = max_source_length, truncation = True)
    labels = tokenizer(targets, max_length = max_target_length, truncation = True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_direction(batch, src_col, tgt_col, prefix, tokenizer, max_source_length, max_target_length):
    """
    Tokenize chỉ một chiều dịch cụ thể 
    Dùng cho tập validation 
    """
    inputs = [prefix + x for x in batch[src_col]]
    targets = batch[tgt_col]

    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs