# BTL_NLP Training Scripts

Các script training này thay thế hoàn toàn các Jupyter notebook, sử dụng framework modular mới với các tính năng được cải thiện.

## 📋 Danh sách Scripts

### Llama3 8B Experiments
- `train_llama3_cnn_dailymail.py` - Fine-tune Llama3-8B trên CNN/DailyMail (summarization)
- `train_llama3_wmt14.py` - Fine-tune Llama3-8B trên WMT14 (translation) 
- `train_llama3_sst2.py` - Fine-tune Llama3-8B trên SST2 (classification)

### Phi-4 Mini Experiments  
- `train_phi4_cnn_dailymail.py` - Fine-tune Phi-4 Mini trên CNN/DailyMail (summarization)
- `train_phi4_wmt14.py` - Fine-tune Phi-4 Mini trên WMT14 (translation)
- `train_phi4_sst2.py` - Fine-tune Phi-4 Mini trên SST2 (classification)

### Llama3 1B Experiments
- `train_llama3_1b_cnn_dailymail.py` - Fine-tune Llama3-1B trên CNN/DailyMail (summarization)
- `train_llama3_1b_wmt14.py` - Fine-tune Llama3-1B trên WMT14 (translation)
- `train_llama3_1b_sst2.py` - Fine-tune Llama3-1B trên SST2 (classification)

### Master Script
- `run_all_experiments.py` - Script chính để chạy tất cả hoặc một số experiments

## 🚀 Cách sử dụng

### 1. Chạy một script đơn lẻ

```bash
cd scripts

# Chạy Llama3 trên CNN/DailyMail
python train_llama3_cnn_dailymail.py

# Chạy Phi-4 trên WMT14
python train_phi4_wmt14.py

# Chạy Llama3-1B trên SST2  
python train_llama3_1b_sst2.py
```

### 2. Sử dụng Master Script

```bash
cd scripts

# Xem danh sách tất cả experiments
python run_all_experiments.py --list

# Chạy tất cả experiments
python run_all_experiments.py --all

# Chạy experiments cụ thể
python run_all_experiments.py llama3_cnn phi4_wmt14

# Chạy tất cả experiments của Llama3
python run_all_experiments.py --model llama3

# Chạy tất cả experiments classification
python run_all_experiments.py --task classification

# Chạy tất cả experiments translation  
python run_all_experiments.py --task translation

# Chạy tất cả experiments summarization
python run_all_experiments.py --task summarization
```

## ⚙️ Cấu hình

Các scripts sử dụng cấu hình được tối ưu hóa cho từng task:

### Summarization (CNN/DailyMail)
- **Learning rate**: 1e-4 (thấp hơn cho summarization)
- **Epochs**: 2
- **LoRA rank**: 16
- **Sequence length**: 2048

### Translation (WMT14)  
- **Learning rate**: 2e-4 (cao hơn cho translation)
- **Epochs**: 3
- **LoRA rank**: 32 (cao hơn cho translation)
- **Warmup steps**: 100

### Classification (SST2)
- **Learning rate**: 3e-4 (cao nhất cho classification)
- **Epochs**: 5 (nhiều hơn cho classification)
- **LoRA rank**: 8 (thấp nhất cho classification)
- **Sequence length**: 512 (ngắn hơn)

## 💾 Kết quả

Tất cả models được lưu trong thư mục `./results/`:

```
results/
├── llama3_8b_cnn_dailymail/      # Llama3-8B CNN/DailyMail results
├── llama3_8b_wmt14/              # Llama3-8B WMT14 results  
├── llama3_8b_sst2/               # Llama3-8B SST2 results
├── phi4_mini_cnn_dailymail/      # Phi-4 CNN/DailyMail results
├── phi4_mini_wmt14/              # Phi-4 WMT14 results
├── phi4_mini_sst2/               # Phi-4 SST2 results
├── llama3_1b_cnn_dailymail/      # Llama3-1B CNN/DailyMail results
├── llama3_1b_wmt14/              # Llama3-1B WMT14 results
└── llama3_1b_sst2/               # Llama3-1B SST2 results
```

Mỗi thư mục chứa:
- **model files** (pytorch_model.bin, config.json, etc.)
- **tokenizer files** 
- **training logs**
- **evaluation results**

## 📊 Yêu cầu hệ thống

| Model | Task | VRAM Required | Estimated Time |
|-------|------|---------------|----------------|
| Llama3-8B | Summarization | 16GB | 2-3 hours |
| Llama3-8B | Translation | 16GB | 3-4 hours |
| Llama3-8B | Classification | 12GB | 1-2 hours |
| Phi-4 Mini | Summarization | 12GB | 1.5-2.5 hours |
| Phi-4 Mini | Translation | 12GB | 2-3 hours |
| Phi-4 Mini | Classification | 8GB | 1-1.5 hours |
| Llama3-1B | Summarization | 8GB | 1-1.5 hours |
| Llama3-1B | Translation | 8GB | 1-2 hours |
| Llama3-1B | Classification | 6GB | 0.5-1 hour |

## 🔧 Tùy chỉnh

Để tùy chỉnh cấu hình training, bạn có thể:

1. **Chỉnh sửa trực tiếp trong script**:
```python
config = TrainingConfig(
    num_train_epochs=5,        # Tăng số epochs
    learning_rate=1e-4,        # Thay đổi learning rate
    per_device_train_batch_size=4,  # Tăng batch size
    max_samples=10000          # Tăng số samples
)
```

2. **Sử dụng config file** (xem `config.yaml` ở root)

3. **Thay đổi model parameters**:
```python
model, tokenizer = load_llama3_model(
    lora_r=32,        # Tăng LoRA rank
    use_dora=True,    # Sử dụng DoRA
    max_seq_length=4096  # Tăng sequence length
)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Giảm `per_device_train_batch_size`
- Tăng `gradient_accumulation_steps`
- Giảm `max_seq_length`
- Sử dụng `fp16=True` thay vì `bf16=True`

### Slow Training
- Tăng `per_device_train_batch_size`
- Giảm `gradient_accumulation_steps`
- Giảm `max_samples` để test nhanh

### Dependencies Issues
- Chạy `pip install -r requirements.txt`
- Đảm bảo CUDA được cài đặt đúng
- Kiểm tra PyTorch version compatibility

## 📈 Monitoring

Scripts tự động hiển thị:
- **Environment info** (GPU, memory, CUDA version)
- **Model parameters** (trainable params, total params)
- **Training progress** (loss, learning rate)
- **Evaluation results** (task-specific metrics)
- **Timing information** (estimated completion time)

## 🎯 So sánh với Notebooks

| Feature | Notebooks | Scripts |
|---------|-----------|---------|
| Code duplication | 100+ lines repeated | 3-5 lines |
| Configuration | Hard-coded | Centralized config |
| Error handling | Manual | Automatic |
| Logging | Minimal | Comprehensive |
| Reproducibility | Variable | Consistent |
| Memory optimization | Manual | Automatic |
| Model comparison | Difficult | Easy |

Scripts mang lại hiệu quả và tính nhất quán cao hơn nhiều so với việc sử dụng notebooks riêng lẻ! 