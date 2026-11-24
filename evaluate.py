"""Evaluate a checkpoint on VisDrone val and report COCO-style mAP.

AP_small is the headline number for this project (objects < 32x32 px).

  python evaluate.py --data /path/to/VisDrone --ckpt checkpoints/msan.pt --name msan
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from data.visdrone import VisDroneDet, collate_fn
from models.detector import load_model


@torch.no_grad()
def evaluate(model, loader, device):
    metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[1, 10, 500])
    model.eval()
    for images, targets in tqdm(loader, desc="evaluating"):
        outputs = model([im.to(device) for im in images])
        preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, gts)
    res = metric.compute()
    keep = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_500"]
    return {k: round(float(res[k]), 4) for k in keep}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True, help="label used for results/<name>.json")
    p.add_argument("--split", default="val")
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = VisDroneDet(args.data, args.split, train=False, max_images=args.max_images)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)
    model = load_model(args.ckpt, device)
    scores = evaluate(model, loader, device)
    print(json.dumps(scores, indent=2))
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.name}.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
