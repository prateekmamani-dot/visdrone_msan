"""Train the baseline or the attention model on VisDrone.

Examples
  python train.py --data /path/to/VisDrone --out checkpoints/msan.pt
  python train.py --data /path/to/VisDrone --no-attention --out checkpoints/baseline.pt
  # ablations
  python train.py --data ... --no-cross   --out checkpoints/abl_no_cross.pt
  python train.py --data ... --no-spatial --out checkpoints/abl_no_spatial.pt
"""
import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.visdrone import VisDroneDet, collate_fn
from models.detector import build_detector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder containing VisDrone2019-DET-* directories")
    p.add_argument("--out", default="checkpoints/msan.pt")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None, help="use a subset (quick debugging)")
    p.add_argument("--no-attention", action="store_true", help="train the plain Faster R-CNN+FPN baseline")
    p.add_argument("--no-cross", action="store_true", help="ablation: remove cross-scale attention")
    p.add_argument("--no-channel", action="store_true", help="ablation: remove channel attention")
    p.add_argument("--no-spatial", action="store_true", help="ablation: remove spatial attention")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found. Training will be very slow; use --max-images for a quick test.")

    ds = VisDroneDet(args.data, "train", train=True, max_images=args.max_images)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_fn, drop_last=True)

    model = build_detector(use_attention=not args.no_attention,
                           cross_scale=not args.no_cross, channel=not args.no_channel,
                           spatial=not args.no_spatial).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    total_steps = args.epochs * len(loader)
    warmup = min(500, len(loader))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for i, (images, targets) in enumerate(bar):
            images = [im.to(device) for im in images]
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch"); opt.zero_grad(); continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
            bar.set_postfix(loss=f"{running / (i + 1):.3f}", lr=f"{opt.param_groups[0]['lr']:.4f}")

        torch.save({"model": model.state_dict(), "use_attention": not args.no_attention,
                    "epoch": epoch + 1}, args.out)
        print(f"saved {args.out} (epoch {epoch + 1})")


if __name__ == "__main__":
    main()
