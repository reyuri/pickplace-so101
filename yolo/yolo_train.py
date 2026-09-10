from ultralytics import YOLO

m = YOLO("/root/autodl-tmp/weights/yolo11n.pt")
r = m.train(
    data="/root/autodl-tmp/yolo_dataset/data.yaml",
    epochs=100,
    imgsz=[480, 640],   # (h, w) 保持源图 640x480 原宽高比
    patience=50,        # 连续50个epoch val mAP 不提升则早停
    batch=16,
    workers=4,
    project="/root/autodl-tmp/yolo_runs",
    name="yolo11n_toy5",
    device=0,
    plots=True,
    verbose=True,
)
print("TRAIN_DONE best_mAP", r.results, flush=True)
