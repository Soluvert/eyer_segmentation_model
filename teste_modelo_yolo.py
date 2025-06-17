import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Carregar modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_unet_path = os.path.join(BASE_DIR, "models/model_eyer_float32.tflite")
model_detector_path = os.path.join(BASE_DIR, "models/best_float32.tflite")

unet_model = tf.lite.Interpreter(model_path=model_unet_path)
detector_model = tf.lite.Interpreter(model_path=model_detector_path)

unet_model.allocate_tensors()
detector_model.allocate_tensors()

# Função para inferir lateralidade com base na posição do disco óptico
def infer_laterality_from_bbox(bbox, image_width=1088):
    x_center = (bbox[0] + bbox[2]) / 2
    if x_center < image_width / 2:
        return "OD (Olho Direito)"
    else:
        return "OE (Olho Esquerdo)"

# Pré-processamento da imagem (canal verde replicado como RGB falso)
def preprocess_image(image_np, size, dtype=np.float32):
    green_channel = image_np[..., 1] / 255.0
    green_rgb = np.stack([green_channel] * 3, axis=-1)  # (H, W, 3)
    resized = cv2.resize(green_rgb, (size, size) if isinstance(size, tuple) else (size, size))
    input_data = np.expand_dims(resized, axis=0).astype(dtype)
    return input_data, resized

# Inferência do modelo de detecção
def run_detection(preprocessed_image):
    input_details = detector_model.get_input_details()
    output_details = detector_model.get_output_details()

    detector_model.set_tensor(input_details[0]['index'], preprocessed_image)
    detector_model.invoke()
    output_data = detector_model.get_tensor(output_details[0]['index'])[0]

    scores = output_data[:, 4]
    best_idx = np.argmax(scores)
    if scores[best_idx] < 0.3:
        raise ValueError("❌ Nenhuma detecção com confiança suficiente.")

    bbox = output_data[best_idx, :4] * 1088
    return bbox

# Recorte quadrado e centralizado
def crop_and_resize(image, bbox, original_size=1088, target_size=256):
    x1, y1, x2, y2 = bbox.astype(int)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    half_size = int(max(x2 - x1, y2 - y1) / 2 * 1.2)

    x1_square = max(0, cx - half_size)
    x2_square = min(original_size, cx + half_size)
    y1_square = max(0, cy - half_size)
    y2_square = min(original_size, cy + half_size)

    cropped = image[y1_square:y2_square, x1_square:x2_square]
    h, w = cropped.shape[:2]
    if h != w:
        side = min(h, w)
        cropped = cropped[0:side, 0:side]

    if cropped.size == 0:
        raise ValueError("❌ Recorte vazio após centralização.")

    resized_crop = cv2.resize(cropped, (target_size, target_size))
    return resized_crop

# Pipeline principal
def run_pipeline(image_path):
    # ⚠️ Lê imagem como RGB (como no treino)
    image_rgb = load_img(image_path, target_size=(1088, 1088))
    image_rgb = img_to_array(image_rgb).astype("uint8")  # (1088, 1088, 3) - RGB

    # Pré-processamento → canal verde expandido para 3 canais
    input_data, normalized_image = preprocess_image(image_rgb, size=1088)

    # Detecção
    bbox = run_detection(input_data)
    print(f"📦 BBox usada (pixels): {bbox}")
    
    # ➕ Inferência da lateralidade
    laterality = infer_laterality_from_bbox(bbox, image_width=1088)
    print(f"👁️ Lateralidade inferida: {laterality}")

    # Recorte na imagem normalizada
    cropped_rgb = crop_and_resize(normalized_image, bbox, original_size=1088, target_size=256)

    # Segmentação → extrai canal verde do recorte e adapta dtype
    input_details = unet_model.get_input_details()
    output_details = unet_model.get_output_details()

    green_channel = cropped_rgb[..., 1]  # (256, 256)
    green_channel = np.expand_dims(green_channel, axis=-1)  # (256, 256, 1)
    input_unet = np.expand_dims(green_channel, axis=0).astype(input_details[0]['dtype'])  # (1, 256, 256, 1)

    # Inferência com U-Net
    unet_model.set_tensor(input_details[0]['index'], input_unet)
    unet_model.invoke()
    pred_mask = unet_model.get_tensor(output_details[0]['index'])[0]

    return image_rgb, cropped_rgb, pred_mask

# Execução principal
if __name__ == "__main__":
    image_path = "/home/andre/Desenvolvimento/SEGMENTATIONOCOD/data/eyer_data_new/Images_Test/0020.png"  # Substitua pelo caminho da sua imagem

    image, cropped, pred_mask = run_pipeline(image_path)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.astype("uint8"))
    axs[0].set_title("Imagem Original")
    axs[0].axis("off")

    axs[1].imshow(cropped)
    axs[1].set_title("Recorte Centralizado")
    axs[1].axis("off")

    axs[2].imshow(np.argmax(pred_mask, axis=-1), cmap="jet")
    axs[2].set_title("Máscara Segmentada")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
