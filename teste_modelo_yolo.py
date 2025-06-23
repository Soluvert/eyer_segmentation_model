import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from scripts.attention_unet import ModelUnetAttention

# === Carregar modelos ===
yolo_model = tf.keras.models.load_model(
    "/home/andre/Desenvolvimento/EYER_ENTREGA/models/model_recorte_eyer.keras"
)
unet_model = tf.keras.models.load_model(
    "/home/andre/Desenvolvimento/EYER_ENTREGA/models/model_eyer.keras"
)

def run_yolo_inference(image, input_size=1088, conf_threshold=0.3, min_size=30):
    image_resized = cv2.resize(image, (input_size, input_size))
    input_tensor = np.expand_dims(image_resized.astype(np.float32) / 255.0, axis=0)

    preds = yolo_model.predict(input_tensor)

    # Verifica estrutura de saída
    if isinstance(preds, dict):
        key = next(iter(preds))  # pega a primeira chave
        preds = preds[key]
        if isinstance(preds, (list, np.ndarray)):
            preds = preds[0]
        else:
            preds = preds.numpy()[0] if hasattr(preds, "numpy") else preds[0]
    elif isinstance(preds, (list, np.ndarray)):
        preds = preds[0]
    else:
        raise TypeError(f"Tipo de saída inesperado: {type(preds)}")

    # Filtro por confiança mínima e tamanho
    valid = []
    for box in preds:
        x, y, w, h = box[:4]
        if w >= min_size and h >= min_size:
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            valid.append([x1, y1, x2, y2])


    if not valid:
        raise ValueError("Nenhuma bbox válida após filtro.")

    best_box = max(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return np.array(best_box), image_resized


# === Pós-processamento ===
def crop_and_resize(image, bbox, target_size=256):
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Recorte vazio após bbox.")

    h_crop, w_crop = crop.shape[:2]
    if h_crop != w_crop:
        side = min(h_crop, w_crop)
        crop = crop[0:side, 0:side]

    return cv2.resize(crop, (target_size, target_size))

def preprocess_for_unet(image_rgb_crop):
    green = image_rgb_crop[..., 1] / 255.0
    green = np.expand_dims(green, axis=-1)
    return np.expand_dims(green, axis=0).astype(np.float32)

# === Pipeline final ===
def run_pipeline(image_path, export_mask=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    bbox, resized = run_yolo_inference(image)
    cropped = crop_and_resize(resized, bbox)
    input_unet = preprocess_for_unet(cropped)
    pred_mask = unet_model.predict(input_unet)[0]

    if export_mask:
        np.save("output_mask.npy", pred_mask)
        mask_png = (np.argmax(pred_mask, axis=-1) * 127).astype(np.uint8)
        cv2.imwrite("output_mask.png", mask_png)

    return resized, cropped, pred_mask

# === Execução ===
if __name__ == "__main__":
    image_path = "/home/andre/Desenvolvimento/SEGMENTATIONOCOD/data/eyer_data_new/Images_Test/0041.png"
    image, cropped, pred_mask = run_pipeline(image_path)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Imagem YOLO")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Recorte pela BBox")
    axs[1].axis("off")

    axs[2].imshow(np.argmax(pred_mask, axis=-1), cmap="jet")
    axs[2].set_title("Máscara U-Net")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig("debug_final_output.png")
    plt.show()
