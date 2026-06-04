# 🎨 StyleFusionArt — Neural Style Transfer

Transform any photo into a stunning artwork using **AdaIN (Adaptive Instance Normalization)** built from scratch with PyTorch.

🔗 **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/meejiya/StyleFusionArt)**

---

## 🤔 What is this?

Neural Style Transfer is a deep learning technique that blends the **content** of one image with the **style** of another — turning your photos into paintings.

This project implements the **AdaIN** method from the paper:
> *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* — Huang & Belongie, 2017

---


## 🖼️ Examples

| Content | Style | Output |
|---|---|---|
| Portrait | Sketch | Stylized Sketch |
| Portrait | Picasso | Stylized Picasso |

---

## 🧠 How it works

1. A **VGG-19 encoder** extracts features from both content and style images
2. **AdaIN** aligns the mean and variance of content features to match the style features
3. A **decoder** reconstructs the stylized image from the transformed features
4. An **alpha** parameter controls the blend between content and style (0 = content only, 1 = full style)

---

## 🗂️ Project Structure

```
StyleFusionArt/
├── utility/
│   ├── flask_app.py       # Flask web app
│   ├── models.py          # VGGEncoder + Decoder
│   ├── utils.py           # AdaIN + dataset utils
│   ├── train.py           # Training script
│   └── templates/
│       └── index.html     # Flask UI
├── examples/              # Sample images
├── app.py                 # Gradio app
└── requirements.txt
```

> 💡 Two web apps included — **Gradio** for the live HuggingFace demo, **Flask** for local deployment with a custom UI.

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch** — model architecture and training
- **Gradio** — web demo
- **Flask** — alternative web app
- **Pillow** — image processing

---

## 🚀 Training

The decoder is trained from scratch on large-scale image datasets. The VGG encoder uses pretrained weights (`vgg_normalised.pth`) and is frozen during training.

**Loss function:**
- Content loss: MSE between decoder output features and AdaIN target
- Style loss: MSE between mean and std of features at multiple VGG layers

```bash
python utility/train.py \
  --content_dir path/to/content \
  --style_dir path/to/style \
  --epochs 2
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/priyanshi11169/StyleFusionArt-.git
cd StyleFusionArt-

# Install dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py
```

> Note: You need `vgg_normalised.pth` and `decoder_final.pth` to run inference.

---

## Built by Me:

**Priyanshi Tiwari**  
