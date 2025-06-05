from ultralytics.models.yolo.yoloe import YOLOESegTrainer
from ultralytics.models.yolo.detect import DetectionTrainer 
from ultralytics import YOLOE

model = YOLOE("pretrained/yoloe-v8l-seg.pt")
head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]
for name, child in model.model.model[-1].named_children():
    if "cv3" not in name:
        freeze.append(f"{head_index}.{name}")

freeze.extend(
    [
        f"{head_index}.cv3.0.0",
        f"{head_index}.cv3.0.1",
        f"{head_index}.cv3.1.0",
        f"{head_index}.cv3.1.1",
        f"{head_index}.cv3.2.0",
        f"{head_index}.cv3.2.1",
    ]
)

model.train(
    data="ultralytics/cfg/datasets/YouYu-JiangYong.yaml",
    epochs=2,
    close_mosaic=0,
    batch=16,
    optimizer="AdamW",
    lr0=1e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=0,
    device="0",
    trainer=YOLOESegTrainer,
    freeze=freeze,
)