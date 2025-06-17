import tensorflow as tf

# Carrega o modelo do novo formato .keras
model = tf.keras.models.load_model("/home/andre/Desenvolvimento/EYER_ENTREGA/models/model_eyer.weights.keras")

# Salva como .h5
model.save("/home/andre/Desenvolvimento/EYER_ENTREGA/models/model_eyer.weights.h5")
