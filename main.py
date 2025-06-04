# Weights and Biases integration for tracking experiments
#import wandb

#wandb.login(key="20213f70b4dfae308af0d7be332b88191ea99653")  # Replace with your actual API key

from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11s-seg.pt")

# Train the model
results = model.train(
                        data="fine-dataset.yaml", 
                        epochs=300, 
                        patience=60, 
                        batch=48, 
                        imgsz=832, 
                        lr0=0.001,
                        lrf=0.01, 
                        save=True, 
                        device=[0,1], 
                        project="yolo11-fine", 
                        name="test-train-11",
                        cos_lr=True, 
                        exist_ok=False, 
                        dropout=0.1, 
                        mixup=0.0, 
                        copy_paste=0.2, 
                        hsv_h=0.02, 
                        hsv_s=0.2, 
                        hsv_v=0.2, 
                        scale=0.7, 
                        degrees=10, 
                        translate=0.3, 
                        shear=30, 
                        perspective=0.001, 
                        flipud=0.15,
                        fliplr=0.15, 
                        mosaic=0.4,
                        warmup_epochs=5,
                        warmup_momentum=0.8,
                        warmup_bias_lr=0.1
                        )
