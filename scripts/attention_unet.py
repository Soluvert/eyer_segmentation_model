import tensorflow as tf
from tensorflow.keras import layers, Model

@tf.keras.utils.register_keras_serializable()
class ModelUnetAttention(tf.keras.Model):
    def __init__(self, input_shape=(512, 512, 1), n_classes=3, base_filters=64, dropout_rate=0.3, **kwargs):
        super(ModelUnetAttention, self).__init__(**kwargs)
        self.input_shape_custom = input_shape
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate

        self.inputs_layer = layers.Input(shape=input_shape)

        def conv_block(x, filters):
            x = layers.SeparableConv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.SeparableConv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x

        def attention_gate(x, g, inter_channels):
            theta_x = layers.Conv2D(inter_channels, (1, 1))(x)
            phi_g = layers.Conv2D(inter_channels, (1, 1))(g)
            if theta_x.shape[1] != phi_g.shape[1] or theta_x.shape[2] != phi_g.shape[2]:
                phi_g = layers.UpSampling2D(size=(theta_x.shape[1] // phi_g.shape[1], theta_x.shape[2] // phi_g.shape[2]))(phi_g)
            add = layers.add([theta_x, phi_g])
            relu = layers.Activation('relu')(add)
            psi = layers.Conv2D(1, (1, 1), activation='sigmoid')(relu)
            return layers.multiply([x, psi])

        # Encoder
        x1 = conv_block(self.inputs_layer, base_filters)
        p1 = layers.MaxPooling2D((2, 2))(x1)

        x2 = conv_block(p1, base_filters*2)
        p2 = layers.MaxPooling2D((2, 2))(x2)

        x3 = conv_block(p2, base_filters*4)
        p3 = layers.MaxPooling2D((2, 2))(x3)

        x4 = conv_block(p3, base_filters*8)
        p4 = layers.MaxPooling2D((2, 2))(x4)

        # Bridge
        x5 = conv_block(p4, base_filters*16)

        # Decoder
        g4 = layers.Conv2D(base_filters*8, (1, 1))(x5)
        att4 = attention_gate(x4, g4, base_filters*8)
        up4 = layers.UpSampling2D((2, 2))(x5)
        up4 = layers.Concatenate()([up4, att4])
        x6 = conv_block(up4, base_filters*8)
        x6 = layers.SpatialDropout2D(dropout_rate)(x6)

        g3 = layers.Conv2D(base_filters*4, (1, 1))(x6)
        att3 = attention_gate(x3, g3, base_filters*4)
        up3 = layers.UpSampling2D((2, 2))(x6)
        up3 = layers.Concatenate()([up3, att3])
        x7 = conv_block(up3, base_filters*4)
        x7 = layers.SpatialDropout2D(dropout_rate)(x7)

        g2 = layers.Conv2D(base_filters*2, (1, 1))(x7)
        att2 = attention_gate(x2, g2, base_filters*2)
        up2 = layers.UpSampling2D((2, 2))(x7)
        up2 = layers.Concatenate()([up2, att2])
        x8 = conv_block(up2, base_filters*2)
        x8 = layers.SpatialDropout2D(dropout_rate)(x8)

        g1 = layers.Conv2D(base_filters, (1, 1))(x8)
        att1 = attention_gate(x1, g1, base_filters)
        up1 = layers.UpSampling2D((2, 2))(x8)
        up1 = layers.Concatenate()([up1, att1])
        x9 = conv_block(up1, base_filters)
        x9 = layers.SpatialDropout2D(dropout_rate)(x9)

        outputs = layers.Conv2D(n_classes, (1, 1), activation="softmax", dtype="float32")(x9)
        self.model = Model(inputs=self.inputs_layer, outputs=outputs)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_custom,
            "n_classes": self.n_classes,
            "base_filters": self.base_filters,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
