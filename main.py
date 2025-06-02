# Weights and Biases integration for tracking experiments
#import wandb

#wandb.login(key="20213f70b4dfae308af0d7be332b88191ea99653")  # Replace with your actual API key

from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11n-seg.pt")

# Train the model
results = model.train(data="fine-dataset.yaml", epochs=100, batch=16, imgsz=640, save=True, device=0, project="yolo11-fine", name="test-train-1", exist_ok=False)
