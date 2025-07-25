import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

# Cargar repo local de YOLOv5
import sys
sys.path.append("yolov5")

# ---------------- CONFIGURACIÓN ----------------
MODELOS = {
    "YOLOv5": "modelos/best.pt",
    "YOLOv8": "modelos/best1.pt"
}
IMG_DIR = "val/images"
LABEL_DIR = "val/labels"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# ---------------- FUNCIONES ----------------
def cargar_labels_yolo(path_txt, img_w, img_h):
    boxes = []
    if not os.path.exists(path_txt):
        return boxes
    with open(path_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            boxes.append([cls, x1, y1, x2, y2])
    return boxes

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

def evaluar_modelo(nombre_modelo, path_modelo):
    print(f"\nEvaluando modelo: {nombre_modelo}")
    y_true = []
    y_pred = []
    TP = 0
    FP = 0
    FN = 0
    ious = []

    if nombre_modelo == "YOLOv5":
        model = torch.hub.load("yolov5", "custom", path=path_modelo, source="local")
    else:
        model = YOLO(path_modelo)

    for file in os.listdir(IMG_DIR):
        if not file.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(IMG_DIR, file)
        label_path = os.path.join(LABEL_DIR, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        gt_boxes = cargar_labels_yolo(label_path, w, h)
        y_true.append(1 if any(b[0] == 0 for b in gt_boxes) else 0)

        if nombre_modelo == "YOLOv5":
            result = model(img)
            pred_boxes = result.pred[0].cpu().numpy()
        else:
            result = model(img_path)
            pred_boxes = result[0].boxes.data.cpu().numpy()

        pred_filtradas = [b for b in pred_boxes if b[4] >= CONF_THRESHOLD and int(b[5]) == 0]
        y_pred.append(1 if len(pred_filtradas) > 0 else 0)

        matched_gt = set()
        for pb in pred_filtradas:
            pred_box = pb[:4]
            pred_box_xy = [pred_box[0], pred_box[1], pred_box[2], pred_box[3]]
            found_match = False
            for i, gt in enumerate(gt_boxes):
                if gt[0] != 0 or i in matched_gt:
                    continue
                iou = calcular_iou(pred_box_xy, gt[1:])
                if iou >= IOU_THRESHOLD:
                    TP += 1
                    ious.append(iou)
                    matched_gt.add(i)
                    found_match = True
                    break
            if not found_match:
                FP += 1
        FN += len([gt for i, gt in enumerate(gt_boxes) if gt[0] == 0 and i not in matched_gt])

    # ---------------- Resultados ----------------
    print("\nMatriz de confusión:")
    matriz = confusion_matrix(y_true, y_pred)
    print(matriz)

    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=["No Arma", "Arma"]))

    print("\n--- Métricas adicionales ---")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"IoU promedio: {np.mean(ious):.4f}" if ious else "IoU promedio: N/A")

    # Graficar matriz
    fig, ax = plt.subplots()
    ax.matshow(matriz, cmap='Blues', alpha=0.7)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(x=j, y=i, s=matriz[i, j], va='center', ha='center')
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.savefig(f"confusion_{nombre_modelo}.png")
    plt.show()

# ---------------- EJECUCIÓN ----------------
if __name__ == "__main__":
    print("Modelos disponibles:")
    for i, name in enumerate(MODELOS.keys()):
        print(f"{i + 1}. {name}")

    idx = int(input("Selecciona el modelo (1 o 2): ")) - 1
    modelo_nombre = list(MODELOS.keys())[idx]
    modelo_path = MODELOS[modelo_nombre]

    evaluar_modelo(modelo_nombre, modelo_path)
# ---------------- FIN ----------------