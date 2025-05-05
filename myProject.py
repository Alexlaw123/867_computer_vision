from ultralytics import YOLO
import os


print("Current working directory:", os.getcwd())

# 1. 加载一个基于 COCO 数据集预训练的 YOLOv8n 模型
# model = YOLO("best.pt")
# model = YOLO("yolov8n_seformer.yaml")
# model = YOLO("yolov8_change.yaml")
model = YOLO("yolov8_transformer_plus_p2.yaml")
# print(model.names)
model.info(verbose=True)


# 2. 显示模型信息（可选）
# model.info()
# model.info(detailed=True, verbose=True)

yaml_path = "/Users/zudongluo/PycharmProjects/pythonProject/YoloV8/bdd100k/bdd100k/bdd100k.yaml"
# 3. 使用 coco8.yaml 这个示例数据集配置文件，训练模型 100 个 epoch，图像大小为 640
# results = model.train(data="coco8.yaml", epochs=10, imgsz=640)
model.train(data=yaml_path, epochs=1, imgsz=640, batch=32, device='mps')  # Use "cuda" if you have a GPU

# metrics = model.val(data=yaml_path, split='val', imgsz=640, batch=32, device='mps')  # 可指定GPU
# print(metrics)

# 4. 用训练好的 YOLOv8n 模型对 'bus.jpg' 进行推理
# results = model("bdd100k/bdd100k/images/10k/val/7d2f7975-e0c1c5a7.jpg",save=True)
# results = model.predict("bdd100k/bdd100k/bdd100k/images/100k/test/",save=True,  device='mps')