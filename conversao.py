import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf

# Caminho do modelo ONNX
onnx_model_path = "/home/andre/Desenvolvimento/SEGMENTATIONOCOD/NEW_YOLO_MODEL/runs/detect/optic_disc_yolo/weights/best.onnx"

# Carrega o modelo ONNX
onnx_model = onnx.load(onnx_model_path)

# Extrai nome das entradas
input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [node.name for node in onnx_model.graph.initializer]
net_feed_input = list(set(input_all) - set(input_initializer))
print(f"Entradas: {net_feed_input}")

# Nome real da entrada
input_name = net_feed_input[0]

# Conversão ONNX → Keras
k_model = onnx_to_keras(
    onnx_model,
    [input_name],
    input_shapes={0: (3, 256, 256)},  # Use índice 0, CHW (Canais, Altura, Largura)
    name_policy='renumerate',
    change_ordering=True
)

# Salva como arquivo .h5 (Keras)
k_model.save("modelo_convertido.h5")
print("✅ Modelo salvo como 'modelo_convertido.h5'")
