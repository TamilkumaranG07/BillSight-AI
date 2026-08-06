from ultralytics import YOLO
import torch
import multiprocessing


def main():

    print("=" * 60)
    print("YOLOv8 BARCODE TRAINING")
    print("=" * 60)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    else:
        print("WARNING: CUDA GPU was not detected.")
        print("Training will run on CPU.")

    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolo11n.pt")

    # Start training
    model.train(
        data=r"C:\Project Phase 1\yolo_dataset\data.yaml",

        epochs=20,

        imgsz=640,

        batch=8,

        device=0,

        # Set to 0 first to avoid Windows multiprocessing errors
        workers=4,

        project=r"C:\Project Phase 1\runs",

        name="barcode_yolov8n",

        exist_ok=True,

        save=True,

        plots=True,

        pretrained=True,

        patience=20,

        seed=42,

        verbose=True
    )

    print("\nTraining completed!")

    print(
        r"Best model: "
        r"C:\Project Phase 1\runs"
        r"\barcode_yolov8n\weights\best.pt"
    )


if __name__ == "__main__":

    # Required for Windows multiprocessing
    multiprocessing.freeze_support()

    main()