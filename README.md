# Acoustic Sensing & Environmental Sound Classification using Optimized AST

An end-to-end, high-efficiency system architecture for classifying urban and environmental acoustic signals. This project leverages the **Audio Spectrogram Transformer (AST)** trained on the **UrbanSound8K** dataset, optimized via **Post-Training Dynamic Quantization (INT8)** for deployment on resource-constrained edge CPU environments.

## 🚀 Key Achievements
* **High Accuracy:** Achieved a baseline classification accuracy of **91%** on complex urban acoustic scenes.
* **Model Compression:** Reduced the model footprint by **~75%** (from ~340MB down to ~85MB) using weight quantization.
* **Hardware Optimized:** Eliminated dependency on high-throughput TPU hardware for inference by adapting the model graph for execution on `QuantizedCPU` backends.

---

## 🛠️ System Architecture & Signal Pipeline

The processing pipeline is split into two distinct stages: Signal Processing (Front-end) and Optimized Transformer Inference (Back-end).

1. **Audio Ingestion & Normalization:** Native audio signals are ingested and dynamically resampled to a standardized **16 kHz** single-channel format using `torchaudio`.
2. **DSP Front-End (Optional Denoising):** Implements **Spectral Subtraction** to estimate and remove stationary background noise profiles before feature extraction.
3. **Spectrogram Generation:** The 1D time-domain signal is converted into a 2D Time-Frequency Mel Spectrogram, segmented into $16 \times 16$ non-overlapping patches.
4. **INT8 Quantized Inference:** Linear attention projections run on an INT8 framework, drastically minimizing CPU memory bandwidth bottlenecks.

---

## 📁 Repository Structure

```text
├── assets/                  # Architecture diagrams and performance plots
├── src/
│   ├── AST_Fine_Tuning_Audio_to_arrow_mapping.ipynb        # Resampling, denoising, and patch extraction
│   ├── AST_Quantization_FineTuned.ipynb          			# Post-training dynamic quantization pipeline
│   └── AST_Fine_Tuning_Training.ipynb 						# Training of the model on new dataset and evaulation results.
├── README.md
└── requirements.txt
```
### Dataset and Fine-tuned model Links
1. **Dataset**: https://www.kaggle.com/datasets/chrisfilo/urbansound8k (7.09GB)
2. **Fine-tuned Model**: https://drive.google.com/drive/folders/18H0WaT6CGagsnEBlAg0O-qYkvk_v_AU2?usp=drive_link
