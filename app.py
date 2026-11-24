"""Gradio demo: upload a drone image and compare baseline vs. attention model.

  python app.py --msan checkpoints/msan.pt --baseline checkpoints/baseline.pt
"""
import argparse

import gradio as gr
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

import config
from models.detector import load_model

COLORS = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93",
          "#ff924c", "#52a675", "#c77dff", "#00b4d8", "#e5989b"]

p = argparse.ArgumentParser()
p.add_argument("--msan", default="checkpoints/msan.pt")
p.add_argument("--baseline", default="checkpoints/baseline.pt")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
msan = load_model(args.msan, device)
try:
    baseline = load_model(args.baseline, device)
except FileNotFoundError:
    baseline = None


@torch.no_grad()
def detect(model, img, thr):
    out = model([F.to_tensor(img).to(device)])[0]
    keep = out["scores"] >= thr
    boxes, labels, scores = out["boxes"][keep].cpu(), out["labels"][keep].cpu(), out["scores"][keep].cpu()
    drawn = img.copy()
    d = ImageDraw.Draw(drawn)
    for b, l, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        d.rectangle(b, outline=COLORS[(l - 1) % 10], width=2)
    return drawn, len(boxes)


def run(image, thr):
    if image is None:
        return None, None, ""
    img = image.convert("RGB")
    a, na = detect(msan, img, thr)
    msg = f"Attention model: {na} detections"
    b = None
    if baseline is not None:
        b, nb = detect(baseline, img, thr)
        msg += f" | Baseline: {nb} detections"
    return a, b, msg


demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="pil", label="Drone image"),
            gr.Slider(0.1, 0.95, value=0.4, step=0.05, label="Confidence threshold")],
    outputs=[gr.Image(label="Multi-scale attention model"), gr.Image(label="Baseline Faster R-CNN"),
             gr.Textbox(label="Summary")],
    title="Small Object Detection in Drone Imagery",
    description="Classes: " + ", ".join(config.CLASS_NAMES),
)

if __name__ == "__main__":
    demo.launch()
