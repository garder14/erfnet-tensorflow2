import tensorflow as tf


class DownsamplerBlock(tf.keras.layers.Layer):
    def __init__(self, ch_in, ch_out):
        super(DownsamplerBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(ch_out - ch_in, kernel_size=(3, 3), strides=2, padding='same')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inp, is_training=True):
        out1 = self.conv(inp)
        out2 = self.pool(inp)
        out = tf.keras.layers.Concatenate(axis=-1)([out1, out2])
        out = self.bn(out, training=is_training)
        return out
        

class NonBottleNeck1D(tf.keras.layers.Layer):
    def __init__(self, ch_out, dropout_rate, dilation_rate):
        super(NonBottleNeck1D, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 1), strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(ch_out, kernel_size=(1, 3), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(dilation_rate, 1))
        self.conv4 = tf.keras.layers.Conv2D(ch_out, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, dilation_rate))
        self.bn2 = tf.keras.layers.BatchNormalization()
        if self.dropout_rate != 0:
            self.drop = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inp, is_training=True):
        out = self.conv1(inp)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn1(out, training=is_training)
        out = tf.nn.relu(out)
        
        out = self.conv3(out)
        out = tf.nn.relu(out)
        out = self.conv4(out)
        out = self.bn2(out, training=is_training)

        if self.dropout_rate != 0:
            out = self.drop(out, training=is_training)
        
        out = tf.keras.layers.Add()([out, inp])
        out = tf.nn.relu(out)

        return out


class UpsamplerBlock(tf.keras.layers.Layer):
    def __init__(self, ch_out):
        super(UpsamplerBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(ch_out, kernel_size=(3, 3), strides=2, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inp, is_training=True):
        out = self.conv(inp)
        out = self.bn(out, training=is_training)
        out = tf.nn.relu(out)
        return out


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.initial_block = DownsamplerBlock(ch_in=3, ch_out=16)

        self.blocks = []
        self.blocks.append(DownsamplerBlock(ch_in=16, ch_out=64))

        for _ in range(5):
           self.blocks.append(NonBottleNeck1D(ch_out=64, dropout_rate=0.03, dilation_rate=1)) 

        self.blocks.append(DownsamplerBlock(ch_in=64, ch_out=128))

        for _ in range(2):
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=2))
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=4))
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=8))
            self.blocks.append(NonBottleNeck1D(ch_out=128, dropout_rate=0.3, dilation_rate=16))
    
    def call(self, inp, is_training=True):
        out = self.initial_block(inp, training=is_training)
        for block in self.blocks:
            out = block(out, training=is_training)
        return out


class Decoder(tf.keras.Model):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.blocks = []
        self.blocks.append(UpsamplerBlock(ch_out=64))
        self.blocks.append(NonBottleNeck1D(ch_out=64, dropout_rate=0, dilation_rate=1))
        self.blocks.append(NonBottleNeck1D(ch_out=64, dropout_rate=0, dilation_rate=1))
        self.blocks.append(UpsamplerBlock(ch_out=16))
        self.blocks.append(NonBottleNeck1D(ch_out=16, dropout_rate=0, dilation_rate=1))
        self.blocks.append(NonBottleNeck1D(ch_out=16, dropout_rate=0, dilation_rate=1))
        self.output_conv = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(2, 2), strides=2, padding='same')

    def call(self, inp, is_training=True):
        out = inp
        for block in self.blocks:
            out = block(out, training=is_training)
        out = self.output_conv(out)
        return out


class ERFNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ERFNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def call(self, inp, is_training=True):
        out = self.encoder(inp, training=is_training)
        out = self.decoder(out, training=is_training)
        return out
