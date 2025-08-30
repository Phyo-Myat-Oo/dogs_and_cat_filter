import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

class UNetModel:
    def __init__(self, input_shape=(128, 128, 3), num_classes=3, use_pretrained=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        
    def conv_block(self, x, filters, kernel_size=3, padding='same', activation='relu'):
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        return x
    
    def encoder_block(self, x, filters):
        x = self.conv_block(x, filters)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x, pool
    
    def decoder_block(self, x, skip_features, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, filters)
        return x
    
    def build_unet_from_scratch(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)
        
        # Bridge
        bridge = self.conv_block(p4, 1024)
        
        # Decoder
        d1 = self.decoder_block(bridge, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)
        
        # Output
        outputs = layers.Conv2D(self.num_classes, 1, padding='same', activation='softmax')(d4)
        
        model = Model(inputs, outputs, name='UNet')
        return model
    
    def build_unet_with_mobilenet(self):
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Use MobileNetV2 as encoder
        base_model = MobileNetV2(
            input_tensor=inputs,
            weights='imagenet',
            include_top=False
        )
        
        # Use specific layers for skip connections
        skip_connection_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu'   # 8x8
        ]
        
        skip_connections = [base_model.get_layer(name).output 
                           for name in skip_connection_names]
        
        # Bottleneck
        bottleneck = base_model.get_layer('out_relu').output
        
        # Decoder
        x = bottleneck
        
        # Upsampling blocks
        upsampling_sizes = [512, 256, 128, 64]
        
        for i, (skip_conn, size) in enumerate(zip(reversed(skip_connections), upsampling_sizes)):
            x = layers.Conv2DTranspose(size, 3, strides=2, padding='same')(x)
            x = layers.Concatenate()([x, skip_conn])
            x = self.conv_block(x, size)
        
        # Final upsampling to original size
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = self.conv_block(x, 64)
        
        # Output layer
        outputs = layers.Conv2D(self.num_classes, 1, padding='same', activation='softmax')(x)
        
        model = Model(inputs, outputs, name='UNet_MobileNet')
        return model
    
    def build_model(self):
        if self.use_pretrained:
            return self.build_unet_with_mobilenet()
        else:
            return self.build_unet_from_scratch()
    
    def compile_model(self, model, learning_rate=1e-4):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                'accuracy',
                tf.keras.metrics.MeanIoU(num_classes=self.num_classes)
            ]
        )
        return model
    
    def dice_coefficient(self, y_true, y_pred, smooth=1):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_coefficient(y_true, y_pred)

def create_model(input_shape=(128, 128, 3), num_classes=3, use_pretrained=True, learning_rate=1e-4):
    unet_builder = UNetModel(input_shape, num_classes, use_pretrained)
    model = unet_builder.build_model()
    model = unet_builder.compile_model(model, learning_rate)
    
    print(f"Model created with input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Using pretrained encoder: {use_pretrained}")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    model.summary()