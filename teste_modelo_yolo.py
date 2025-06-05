import cv2
import numpy as np
import tensorflow as tf
from scripts.attention_unet import ModelUnetAttention

# Carregar modelos
interpreter = tf.lite.Interpreter(model_path="/home/andre/Desenvolvimento/SEGMENTATIONOCOD/NEW_YOLO_MODEL/runs/detect/optic_disc_yolo/weights/best_saved_model/best_float32.tflite")
interpreter.allocate_tensors()

unet_model = tf.lite.Interpreter(model_path="/home/andre/Desenvolvimento/SEGMENTATIONOCOD/models/model_eyer_float32.tflite")
unet_model.allocate_tensors()

# Pré-processamento para o YOLO
def preprocess_yolo(image, size=1088):
    resized = cv2.resize(image, (size, size))
    input_data = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data, resized

# Inferência do YOLO e conversão da bbox para pixels
def run_yolo_inference(image):
    input_data, resized = preprocess_yolo(image)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    scores = output_data[:, 4]
    best_idx = np.argmax(scores)
    if scores[best_idx] < 0.3:
        raise ValueError("❌ Nenhuma detecção com confiança suficiente.")

    bbox_normalized = output_data[best_idx, :4]
    bbox_pixels = bbox_normalized * 1088
    return bbox_pixels, resized

# Recorte quadrado e centralizado com bbox em pixels
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

    # Garantir corte quadrado se necessário
    h, w = cropped.shape[:2]
    if h != w:
        side = min(h, w)
        cropped = cropped[0:side, 0:side]

    if cropped.size == 0:
        raise ValueError("❌ Recorte vazio após centralização.")

    resized_crop = cv2.resize(cropped, (target_size, target_size))
    return resized_crop

# Pré-processamento do recorte: canal verde → normalizado → (1, 256, 256, 1)
def preprocess_for_unet(image_rgb_crop):
    green_channel = image_rgb_crop[..., 1] / 255.0
    green_channel = np.expand_dims(green_channel, axis=-1)  # (256, 256, 1)
    return np.expand_dims(green_channel, axis=0).astype(np.float32)  # (1, 256, 256, 1)

# Pipeline completo
def run_pipeline(image_path):
    # Carregar e redimensionar a imagem
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Imagem não encontrada: {image_path}")
    
    image = cv2.resize(image, (1088, 1088))  # Compatível com YOLO

    # Inferência YOLO para obter bbox
    bbox, _ = run_yolo_inference(image)
    print(f"📦 BBox usada (pixels): {bbox}")

    # Recorte e redimensionamento
    cropped = crop_and_resize(image, bbox)

    # Pré-processamento para U-Net
    input_unet = preprocess_for_unet(cropped)

    # Inferência com TFLite (U-Net)
    input_details = unet_model.get_input_details()
    output_details = unet_model.get_output_details()

    # Certificar dtype
    # Certificar dtype
    input_unet = input_unet.astype(input_details[0]['dtype'])  # ✅ CORRETO


    # Executar inferência
    unet_model.set_tensor(input_details[0]['index'], input_unet)
    unet_model.invoke()
    pred_mask = unet_model.get_tensor(output_details[0]['index'])[0]

    return image, cropped, pred_mask


if __name__ == "__main__":
    image_path = "/home/andre/Desenvolvimento/SEGMENTATIONOCOD/data/eyer_data_new/Images_Test/0020.png"

    image, cropped, pred_mask = run_pipeline(image_path)

    # Visualização lado a lado
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Imagem Original")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Recorte Centralizado")
    axs[1].axis("off")

    axs[2].imshow(np.argmax(pred_mask, axis=-1), cmap="jet")
    axs[2].set_title("Máscara U-Net (argmax)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
