# 🔬 Microplastics Detector
### MobileNetV3 + SSDLite320 Object Detection

---

## Dataset
- **Train**: 577 images | 5,425 bounding box annotations
- **Valid**: 204 images | 1,701 annotations
- **Class**: `Microplastic` (single class)
- **Image size**: ~563×537 px
- **Avg particles per image**: ~9.4

---

## Project Structure

```
microplastics_detector/
├── dataset.py       # Dataset class, transforms, collate_fn
├── model.py         # SSDLite320 + MobileNetV3 model builder
├── train.py         # Single-epoch training loop
├── evaluate.py      # Validation loss computation
├── inference.py     # Single-image prediction + annotated output
├── export.py        # Export trained model to ONNX
├── utils.py         # Dataset extraction + loss curve plotter
├── main.py          # Entry point: orchestrates full training run
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Extract Your Dataset

```bash
python utils.py --extract_zip archive__1_.zip --data_dir dataset
```

This creates:
```
dataset/
  train/
    _annotations.csv
    *.jpg
  valid/
    _annotations.csv
    *.jpg
```

---

## Step 2 — Train the Model

```bash
# Default: 30 epochs, batch size 8, lr=0.01
python main.py --data_dir dataset --epochs 30 --batch_size 8

# Extract + train in one command:
python main.py --extract_zip archive__1_.zip --data_dir dataset --epochs 30

# Lower batch size if you run out of GPU memory:
python main.py --data_dir dataset --epochs 30 --batch_size 4
```

**Outputs after training:**
- `best_model.pth` — best weights (lowest validation loss)
- `last_model.pth` — final epoch weights
- `training_history.json` — loss values per epoch

---

## Step 3 — Plot Loss Curves

```bash
python utils.py --plot --history training_history.json
# Saves loss_curve.png
```

---

## Step 4 — Run Inference

```bash
python inference.py --image path/to/water_sample.jpg --weights best_model.pth --conf_thresh 0.5
```

**Output:**
- `inference_result.jpg` — image with red bounding boxes drawn around each microplastic particle
- Console: count of detections and per-box confidence scores

---

## Step 5 — Export to ONNX (mobile/edge)

```bash
python export.py --weights best_model.pth --output microplastics.onnx
```

---

## Model Architecture

| Component | Details |
|---|---|
| Backbone | MobileNetV3-Large (pretrained on ImageNet) |
| Detection Head | SSDLite320 |
| Input Size | 320×320 px |
| Classes | background + Microplastic |
| Pretrained | COCO (fine-tuned on your dataset) |

---

## Tips

- **Low GPU memory?** Use `--batch_size 4` or `--batch_size 2`
- **Overfitting?** Lower `--lr 0.005` or add augmentation in `dataset.py → get_transforms()`
- **More accuracy?** Increase `--epochs 50` or `60`
- **Catch more particles?** Lower `--conf_thresh 0.3` (may increase false positives)
