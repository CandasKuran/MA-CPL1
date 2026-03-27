import os
import sys
import json
from ultralytics import YOLO

def detect_objects(image_path):
    model = YOLO("yolov8n.pt")
    results = model(image_path, save=True)

    result = results[0]
    output_dir = result.save_dir
    detections = []

    boxes = result.boxes
    names = result.names

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "class_name": names[cls_id],
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    json_path = os.path.join(output_dir, "detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=4, ensure_ascii=False)

    print("Résultats enregistrés dans :", output_dir)
    print("Fichier JSON créé :", json_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python detect_yolo.py attachments/objet.jpg")
        sys.exit(1)

    detect_objects(sys.argv[1])