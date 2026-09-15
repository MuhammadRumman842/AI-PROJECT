"""Visualize inference predictions on a VLMOD image.

This is an engineering/demo visualization for the reconstructed CyclopsNet-inspired
baseline. It can optionally overlay ground-truth 2D boxes from a matched train JSON.
It does not compute official challenge metrics.
"""
from __future__ import annotations
import argparse, ast, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer import load_model, run as run_inference, device as get_device


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/stage11_controlled/best_val.pt')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/visualization.png')
    p.add_argument('--threshold', type=float, default=0.30)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--gt-json', type=Path, default=None)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    return p.parse_args()


def gt_boxes(json_path: Path):
    if json_path is None:
        return []
    data = json.loads(json_path.read_text(encoding='utf-8'))
    out=[]
    for group in data if isinstance(data,list) else []:
        anns = group if isinstance(group, list) else [group]
        for ann in anns:
            for raw in ann.get('label_3', []):
                try:
                    vals=ast.literal_eval(raw)
                # [class, instance, group, depth, xmin, ymin, xmax, ymax, ...]
                    if len(vals) >= 8:
                        out.append({'class': str(vals[0]), 'box':[float(vals[4]),float(vals[5]),float(vals[6]),float(vals[7])]})
                except (ValueError, SyntaxError, TypeError):
                    continue
    return out


def draw(img, preds, gt):
    d=ImageDraw.Draw(img)
    W,H=img.size
    for g in gt:
        x1,y1,x2,y2=g['box']
        d.rectangle((x1,y1,x2,y2), outline='blue', width=max(2, W//700))
        d.text((x1,max(0,y1-18)), f"GT {g['class']}", fill='blue')
    for p in preds:
        x1,y1,x2,y2=p['bbox_2d_normalized_xyxy']
        box=(x1*W,y1*H,x2*W,y2*H)
        label=f"P {p['class']} s={p['score']:.2f} sim={p['query_similarity']:.2f}"
        d.rectangle(box, outline='red', width=max(2, W//500))
        d.text((box[0],min(H-18,max(0,box[1]-18))), label, fill='red')
    return img


def main():
    a=args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    dev=get_device(a.device)
    model,_=load_model(a.checkpoint, dev)
    result=run_inference(a.image,a.query,model,dev,a.threshold,a.top_k)
    gt=gt_boxes(a.gt_json)
    with Image.open(a.image) as im:
        canvas=im.convert('RGB').copy()
    draw(canvas,result['predictions'],gt).save(a.output)
    sidecar=a.output.with_suffix('.json')
    sidecar.write_text(json.dumps({'query':a.query,'prediction_count':len(result['predictions']),'ground_truth_box_count':len(gt),'predictions':result['predictions']},indent=2),encoding='utf-8')
    print('STAGE 13 VISUALIZATION: OK')
    print('image:', a.output)
    print('json:', sidecar)
    print('predictions:', len(result['predictions']), 'ground_truth_boxes:', len(gt))

if __name__=='__main__': main()
