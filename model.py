import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class BinaryShapeEncoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BinaryShapeEncoder, self).__init__(**kwargs)

        self.conv1 = layers.Conv3D(64, 4, (2, 2, 2), padding='same')
        self.conv2 = layers.Conv3D(128, 4, (2, 2, 2), padding='same')
        self.conv3 = layers.Conv3D(256, 4, (2, 2, 2), padding='same')
        self.conv4 = layers.Conv3D(256, 4, (1, 1, 1), padding='valid')

        self.act1 = layers.LeakyReLU()
        self.act2 = layers.LeakyReLU()
        self.act3 = layers.LeakyReLU()
        self.act4 = layers.LeakyReLU()

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()

        self.flat = layers.Flatten()

    def call(self, inputs, training=False):

        # inputs should be in the shape of (B, H, W, D, C)
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x, training=training)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn4(x, training=training)

        outputs = self.flat(x)

        return outputs


class Projection(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Projection, self).__init__(**kwargs)
        self.fc = layers.Dense(256, use_bias=False)

    def call(self, inputs):
        outputs = self.fc(inputs)
        return outputs


class Decomposer(keras.layers.Layer):

    def __init__(self, num_parts, **kwargs):
        super(Decomposer, self).__init__(**kwargs)
        self.projection_layer_list = list()
        self.binary_shape_encoder = BinaryShapeEncoder()
        for i in range(num_parts):
            self.projection_layer_list.append(Projection())

    def call(self, inputs, training=False):
        projection_layer_outputs = list()
        x = self.binary_shape_encoder(inputs, training=training)
        for each_layer in self.projection_layer_list:
            projection_layer_outputs.append(each_layer(x))
        # outputs should be in the shape of (B, num_parts, encoding_dimensions)
        outputs = tf.transpose(projection_layer_outputs, (1, 0, 2))
        return outputs


class SharedPartDecoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SharedPartDecoder, self).__init__(**kwargs)

        # inputs should be in the shape of (batch_size, num_dimension)
        self.reshape = layers.Reshape((1, 1, 1, 256))

        # inputs should be in the shape of (B, D, H, W, C)
        self.deconv1 = layers.Conv3DTranspose(256, 4, (1, 1, 1), padding='valid')
        self.deconv2 = layers.Conv3DTranspose(128, 4, (2, 2, 2), padding='same')
        self.deconv3 = layers.Conv3DTranspose(64, 4, (2, 2, 2), padding='same')
        self.deconv4 = layers.Conv3DTranspose(1, 4, (2, 2, 2), padding='same', activation='sigmoid')

        self.act = layers.LeakyReLU()
        self.act1 = layers.LeakyReLU()
        self.act2 = layers.LeakyReLU()
        self.act3 = layers.LeakyReLU()

        self.bn = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()

        self.dropout = layers.Dropout(0.2)
        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.2)
        self.dropout3 = layers.Dropout(0.2)

    def call(self, inputs, training=False):

        # inputs should be in the shape of (batch_size, num_dimension)
        x = self.reshape(inputs)
        self.out1 = x

        x = self.deconv1(x)
        self.out2 = self.act1(x)
        x = self.bn1(self.out2, training=training)
        x = self.dropout1(x, training=training)

        x = self.deconv2(x)
        self.out3 = self.act2(x)
        x = self.bn2(self.out3, training=training)
        x = self.dropout2(x, training=training)

        x = self.deconv3(x)
        self.out4 = self.act3(x)
        x = self.bn3(self.out4, training=training)
        x = self.dropout3(x, training=training)

        outputs = self.deconv4(x)

        return outputs


class LocalizationNet(keras.layers.Layer):

    def __init__(self, num_parts, **kwargs):
        super(LocalizationNet, self).__init__(**kwargs)

        # the shape of the stacked input should be (B, num_parts, H, W, D, C)
        # the shape of the summed input should be (B, encoding_dimensions)
        self.stacked_flat = layers.Flatten()

        self.stacked_fc1 = layers.Dense(256)
        self.stacked_fc2 = layers.Dense(256)
        self.summed_fc1 = layers.Dense(128)
        self.final_fc1 = layers.Dense(128)
        self.final_fc2 = layers.Dense(num_parts*12)

        self.stacked_act1 = layers.LeakyReLU()
        self.stacked_act2 = layers.LeakyReLU()
        self.summed_act1 = layers.LeakyReLU()
        self.final_act1 = layers.LeakyReLU()

        self.stacked_dropout1 = layers.Dropout(0.3)
        self.stacked_dropout2 = layers.Dropout(0.3)
        self.summed_dropout1 = layers.Dropout(0.3)
        self.final_dropout1 = layers.Dropout(0.3)

    def call(self, inputs, training=False):

        stacked_inputs = inputs[0]
        summed_inputs = inputs[1]

        # processing stacked inputs
        stacked_x = self.stacked_flat(stacked_inputs)
        stacked_x = self.stacked_fc1(stacked_x)
        stacked_x = self.stacked_act1(stacked_x)
        stacked_x = self.stacked_dropout1(stacked_x, training=training)
        stacked_x = self.stacked_fc2(stacked_x)
        stacked_x = self.stacked_act2(stacked_x)
        stacked_x = self.stacked_dropout2(stacked_x, training=training)

        # processing summed inputs
        summed_x = self.summed_fc1(summed_inputs)
        summed_x = self.summed_act1(summed_x)
        summed_x = self.summed_dropout1(summed_x, training=training)

        # concatenate stacked inputs and summed inputs into final inputs
        final_x = tf.concat([stacked_x, summed_x], axis=1)
        final_x = self.final_fc1(final_x)
        final_x = self.final_act1(final_x)
        final_x = self.final_dropout1(final_x, training=training)
        final_x = self.final_fc2(final_x)

        # the shape of outputs should be (B, num_parts, 3, 4)
        outputs = tf.reshape(final_x, (stacked_inputs.shape[0], stacked_inputs.shape[1], 3, 4))
        return outputs


class Resampling(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Resampling, self).__init__(**kwargs)

    def call(self, inputs):

        # input_fmap has shape (B, num_parts, H, W, D, C)
        # theta has shape (B, num_parts, 3, 4)
        input_fmap = inputs[0]
        theta = inputs[1]

        # batch_grids has shape (B, num_parts, 3, H, W, D)
        batch_grids = self._affine_grid_generator(input_fmap, theta)

        # x_s, y_s and z_s have shape (B, num_parts, H, W, D)
        y_s = batch_grids[:, :, 0, :, :, :]
        x_s = batch_grids[:, :, 1, :, :, :]
        z_s = batch_grids[:, :, 2, :, :, :]

        # output_fmap has shape (B, num_parts, H, W, D, C)
        output_fmap = self._trilinear_sampler(input_fmap, x_s, y_s, z_s)

        return output_fmap

    @staticmethod
    def _affine_grid_generator(input_fmap, theta):
        """
        :param input_fmap: the stacked decoded parts in the shape of (B, num_parts, H, W, D, C)
        :param theta: the output of LocalizationNet. has shape (B, num_parts, 3, 4)
        :return: affine grid for the input feature map. affine grid has shape (B, num_parts, 3, H, W, D)
        """

        # get B, num_parts, H, W, D
        B = tf.shape(input_fmap)[0]
        num_parts = tf.shape(input_fmap)[1]
        H = tf.shape(input_fmap)[2]
        W = tf.shape(input_fmap)[3]
        D = tf.shape(input_fmap)[4]

        # create 3D grid, which are the x, y, z coordinates of the output feature map
        x = tf.cast(tf.linspace(0, W-1, W), dtype=tf.uint8)
        y = tf.cast(tf.linspace(0, H-1, H), dtype=tf.uint8)
        z = tf.cast(tf.linspace(0, D-1, D), dtype=tf.uint8)
        x_t, y_t, z_t = tf.meshgrid(x, y, z)

        # flatten every x, y, z coordinates
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])
        z_t_flat = tf.reshape(z_t, [-1])
        # x_t_flat has shape (H*W*D,)

        # reshape to (x_t, y_t, z_t, 1), which is homogeneous form
        ones = tf.ones_like(x_t_flat, dtype=tf.uint8)
        sampling_grid = tf.stack([y_t_flat, x_t_flat, z_t_flat, ones])
        # sampling_grid now has shape (4, H*W*D)

        # repeat the grid num_batch times along axis=0 and num_parts times along axis=1
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.expand_dims(sampling_grid, axis=1)
        sampling_grid = tf.tile(sampling_grid, [B, num_parts, 1, 1])
        # sampling grid now has shape (B, num_parts, 4, H*W*D)

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid using batch multiply
        batch_grids = theta @ sampling_grid
        # batch grid now has shape (B, num_parts, 3, H*W*D)

        # reshape to (B, num_parts, 3, H, W, D)
        batch_grids = tf.reshape(batch_grids, [B, num_parts, 3, H, W, D])

        return batch_grids

    def _trilinear_sampler(self, input_fmap, x, y, z):

        """
        :param input_fmap: the stacked decoded parts in the shape of (B, num_parts, H, W, D, C)
        :param x: x coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :param y: y coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :param z: z coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :return: interpolated volume in the shape of (B, num_parts, H, W, D, C)
        """

        # grab 8 nearest corner points for each (x_i, y_i, z_i) in input_fmap
        # 2*2*2 combination, so that there are 8 corner points in total
        B, num_parts, H, W, D, C = input_fmap.shape
        temp = tf.Variable(tf.zeros([B, num_parts, H+4, W+4, D+4, C]), dtype='float32')[:, :, 2:2+H, 2:2+W, 2:2+D, :].assign(input_fmap)
        input_fmap_new = tf.constant(temp)

        x = x + 2
        y = y + 2
        z = z + 2

        x0 = tf.cast(tf.floor(x), 'int32')
        y0 = tf.cast(tf.floor(y), 'int32')
        z0 = tf.cast(tf.floor(z), 'int32')
        x0 = tf.clip_by_value(x0, 0, H + 2)
        y0 = tf.clip_by_value(y0, 0, H + 2)
        z0 = tf.clip_by_value(z0, 0, H + 2)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # get voxel value of corner coordinates
        # reference: https://blog.csdn.net/webzhuce/article/details/86585489
        c000 = self._get_voxel_value(input_fmap_new, x0, y0, z0)
        c001 = self._get_voxel_value(input_fmap_new, x0, y0, z1)
        c010 = self._get_voxel_value(input_fmap_new, x0, y1, z0)
        c011 = self._get_voxel_value(input_fmap_new, x0, y1, z1)
        c100 = self._get_voxel_value(input_fmap_new, x1, y0, z0)
        c101 = self._get_voxel_value(input_fmap_new, x1, y0, z1)
        c110 = self._get_voxel_value(input_fmap_new, x1, y1, z0)
        c111 = self._get_voxel_value(input_fmap_new, x1, y1, z1)
        # cxxx has shape (B, num_parts, H, W, D, C)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        y0 = tf.cast(y0, 'float32')
        z0 = tf.cast(z0, 'float32')

        # calculate deltas
        xd = x - x0
        yd = y - y0
        zd = z - z0

        # compute output (this is the  trilinear interpolation formula in real world coordinate system)
        C = tf.shape(input_fmap)[5]
        xd = tf.expand_dims(xd, axis=5)
        yd = tf.expand_dims(yd, axis=5)
        zd = tf.expand_dims(zd, axis=5)
        xd = tf.tile(xd, [1, 1, 1, 1, 1, C])
        yd = tf.tile(yd, [1, 1, 1, 1, 1, C])
        zd = tf.tile(zd, [1, 1, 1, 1, 1, C])
        output_fmap = c000 * (1 - xd) * (1 - yd) * (1 - zd) + c100 * xd * (1 - yd) * (1 - zd) + \
                      c010 * (1 - xd) * yd * (1 - zd) + c001 * (1 - xd) * (1 - yd) * zd + \
                      c101 * xd * (1 - yd) * zd + c011 * (1 - xd) * yd * zd + \
                      c110 * xd * yd * (1 - zd) + c111 * xd * yd * zd
        # output_fmap has shape (B, num_parts, H, W, D, C)

        return output_fmap

    @staticmethod
    def _get_voxel_value(input_fmap, x, y, z):

        """
        :param input_fmap: input feature map in the shape of (B, num_parts, H, W, D, C)
        :param x: x coordinates in the shape of (B, num_parts, H, W, D)
        :param y: y coordinates in the shape of (B, num_parts, H, W, D)
        :param z: z coordinates in the shape of (B, num_parts, H, W, D)
        :return: voxel value in the shape of (B, num_parts, H, W, D, C)
        """

        # pay attention to the difference of ordering between tensor indexing and voxel coordinates
        indices = tf.stack([y, x, z], axis=5)
        return tf.gather_nd(input_fmap, indices, batch_dims=2)


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, num_heads, d_model, keep_channel, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.d_model = d_model
        self.keep_channel = keep_channel

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def call(self, inputs):
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scaled_attention = self._scaled_dot_product_attention(q, k, v)
        if self.keep_channel:
            # scaled_attention.shape == (batch_size, channel, seq_len, num_heads, depth)
            scaled_attention = tf.transpose(scaled_attention, (0, 1, 3, 2, 4))
            # concat_attention.shape == (batch_size, channel, seq_len, d_model)
            concat_attention = tf.reshape(scaled_attention, (tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.d_model))
        else:
            # scaled_attention.shape == (batch_size, seq_len, num_heads, depth)
            scaled_attention = tf.transpose(scaled_attention, (0, 2, 1, 3))
            # concat_attention.shape == (batch_size, seq_len, d_model)
            concat_attention = tf.reshape(scaled_attention, (tf.shape(inputs)[0], tf.shape(inputs)[1], self.d_model))
        outputs = self.dense(concat_attention)
        return outputs

    @staticmethod
    def _scaled_dot_product_attention(q, k, v):
        # q.shape == k.shape == (batch_size, num_heads, seq_len, depth)
        # matmul_qk == (batch_size, num_heads, seq_len, seq_len)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # softmax is normalized on the last axis so that the scores add up to 1.
        # attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # outputs.shape == (batch_size, num_heads, seq_len, depth)
        outputs = tf.matmul(attention_weights, v)
        return outputs

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        if self.keep_channel:
            x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.num_heads, self.depth))
            x = tf.transpose(x, (0, 1, 3, 2, 4))
        else:
            x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.num_heads, self.depth))
            x = tf.transpose(x, (0, 2, 1, 3))
        return x


class FeedForward(keras.layers.Layer):

    def __init__(self, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = keras.layers.Dense(1024, activation='relu')
        self.dense2 = keras.layers.Dense(d_model)

    def call(self, inputs):
        # x.shape == (batch_size, seq_len, 256)
        x = self.dense1(inputs)
        # outputs.shape == (batch_size, seq_len, d_model)
        outputs = self.dense2(x)
        return outputs


class AttentionBlock(keras.layers.Layer):

    def __init__(self, num_heads, d_model, keep_channel, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads, d_model, keep_channel)
        self.ff = FeedForward(d_model)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(0.2)
        self.dropout2 = keras.layers.Dropout(0.2)

    def call(self, inputs, training=False):
        x = self.mha(inputs)
        x = self.dropout1(x, training=training)
        outputs1 = self.layer_norm1(x + inputs, training=training)
        x = self.ff(outputs1)
        x = self.dropout2(x, training=training)
        # outputs2.shape == (batch_size, seq_len, d_model)
        outputs2 = self.layer_norm2(x + outputs1, training=training)
        return outputs2


class AttentionLayer(keras.layers.Layer):

    def __init__(self, num_blocks, num_heads, d_model, seq_len, keep_channel, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self.keep_channel = keep_channel
        self.positional_encoding = self._get_positional_encoding(seq_len, d_model, keep_channel)
        self.attention_blocks = [AttentionBlock(num_heads, d_model, keep_channel) for _ in range(num_blocks)]
        self.dense = layers.Dense(d_model)

    def call(self, inputs, training=False):
        if self.keep_channel:
            if inputs.ndim == 6:
                inputs = tf.transpose(inputs, (0, 5, 1, 2, 3, 4))
                inputs = tf.reshape(inputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], -1))
            else:
                inputs = tf.expand_dims(inputs, axis=1)
        else:
            if inputs.ndim == 6:
                inputs = tf.reshape(inputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], -1))
        x = self.dense(inputs)
        # positional_encoding.shape == (1, seq_len, d_model)
        x = x + self.positional_encoding
        for each_block in self.attention_blocks:
            x = each_block(x, training=training)
        return x

    @staticmethod
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _get_positional_encoding(self, seq_len, d_model, keep_channel):
        angle_rads = self._get_angles(np.arange(seq_len)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, np.newaxis, ...] if keep_channel else angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)


class Model(keras.Model):

    def __init__(self, num_parts, bce_weight, bce_weight_shape, training_process, use_attention, keep_channel, use_ac_loss, which_layer, num_blocks, num_heads, d_model, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.num_parts = num_parts
        self.bce_weight = bce_weight
        self.bce_weight_shape = bce_weight_shape
        self.training_process = training_process
        self.use_attention = use_attention
        self.keep_channel = keep_channel
        self.use_ac_loss = use_ac_loss
        self.which_layer = which_layer

        # build layers
        self.decomposer = Decomposer(num_parts)

        if use_attention:
            if training_process == 1 or training_process == '1':
                self.part_decoder = SharedPartDecoder()
            elif training_process == 2 or training_process == '2':
                self.part_decoder = SharedPartDecoder()
                self.attention_layer_list = [AttentionLayer(num_blocks, num_heads, d_model, num_parts, keep_channel) for i in range(len(which_layer))]
                if keep_channel:
                    self.conv_list = [layers.Conv1D(1, 1, 1, padding='valid') for i in range(len(which_layer))]
                self.dense = layers.Dense(12)
            else:
                self.part_decoder = SharedPartDecoder()
                self.attention_layer_list = [AttentionLayer(num_blocks, num_heads, d_model, num_parts, keep_channel) for i in range(len(which_layer))]
                if keep_channel:
                    self.conv_list = [layers.Conv1D(1, 1, 1, padding='valid') for i in range(len(which_layer))]
                self.dense = layers.Dense(12)
                self.resampling = Resampling()
        else:
            if training_process == 1 or training_process == '1':
                self.part_decoder = SharedPartDecoder()
            elif training_process == 2 or training_process == '2':
                self.part_decoder = SharedPartDecoder()
                self.localization_net = LocalizationNet(num_parts)
            else:
                self.part_decoder = SharedPartDecoder()
                self.localization_net = LocalizationNet(num_parts)
                self.resampling = Resampling()

        # create some loss tracker
        self.pi_loss_tracker = tf.keras.metrics.Mean()
        self.part_reconstruction_loss_tracker = tf.keras.metrics.Mean()
        self.transformation_loss_tracker = tf.keras.metrics.Mean()
        self.ac_loss_tracker = tf.keras.metrics.Mean()
        self.shape_reconstruction_loss_tracker = tf.keras.metrics.Mean()
        self.total_loss_tracker = tf.keras.metrics.Mean()

        # create some evaluation tracker
        self.transformation_mse_tracker = tf.keras.metrics.Mean()
        self.part_mIoU_tracker_list = [tf.keras.metrics.MeanIoU(2) for i in range(num_parts)]
        self.all_part_mIoU_tracker = tf.keras.metrics.MeanIoU(2)
        self.shape_mIoU_tracker = tf.keras.metrics.Mean(2)

    def call(self, inputs, training=False, decomposer_output=None):

        if self.use_attention:
            if self.training_process == 1 or self.training_process == '1':
                # decomposer output has shape (B, num_parts, encoding_dims)
                if decomposer_output is None:
                    decomposer_output = self.decomposer(inputs, training=training)
                decoder_outputs = list()
                decoder_inputs = tf.transpose(decomposer_output, (1, 0, 2))
                for each in decoder_inputs:
                    decoder_outputs.append(self.part_decoder(each, training=training))
                # stacked_decoded_parts should be in the shape of (B, num_parts, H, W, D, C)
                self.stacked_decoded_parts = tf.stack(decoder_outputs, axis=1)
                return self.stacked_decoded_parts

            elif self.training_process == 2 or self.training_process == '2':
                if decomposer_output is None:
                    decomposer_output = self.decomposer(inputs, training=training)
                out1, out2, out3, out4, decoder_outputs = [], [], [], [], []
                decoder_inputs = tf.transpose(decomposer_output, (1, 0, 2))
                for each in decoder_inputs:
                    decoder_output = self.part_decoder(each, training=training)
                    out1.append(self.part_decoder.out1)
                    out2.append(self.part_decoder.out2)
                    out3.append(self.part_decoder.out3)
                    out4.append(self.part_decoder.out4)
                    decoder_outputs.append(decoder_output)
                # stacked_decoded_parts should be in the shape of (B, num_parts, H, W, D, C)
                inter1 = tf.stack(out1, axis=1)
                inter2 = tf.stack(out2, axis=1)
                inter3 = tf.stack(out3, axis=1)
                inter4 = tf.stack(out4, axis=1)
                self.stacked_decoded_parts = tf.stack(decoder_outputs, axis=1)
                self.attention_output_list = list()
                for i, each_attention in enumerate(self.attention_layer_list):
                    if self.which_layer[i] is '0':
                        self.attention_output_list.append(each_attention(decomposer_output, training=training))
                    elif self.which_layer[i] is '1':
                        self.attention_output_list.append(each_attention(inter1, training=training))
                    elif self.which_layer[i] is '2':
                        self.attention_output_list.append(each_attention(inter2, training=training))
                    elif self.which_layer[i] is '3':
                        self.attention_output_list.append(each_attention(inter3, training=training))
                    elif self.which_layer[i] is '4':
                        self.attention_output_list.append(each_attention(inter4, training=training))
                    elif self.which_layer[i] is '5':
                        self.attention_output_list.append(each_attention(self.stacked_decoded_parts, training=training))
                    else:
                        raise ValueError('which_layer should be one or more of 0, 1, 2, 3, 4, and 5')
                if self.keep_channel:
                    self.temp = list()
                    for each, conv in zip(self.attention_output_list, self.conv_list):
                        each = tf.transpose(each, (0, 2, 3, 1))  # channel is at last dimension
                        each = tf.reshape(conv(each), (each.shape[0], each.shape[1], -1))  # 1D convolution
                        self.temp.append(each)
                    concat_output = tf.concat(self.temp, axis=2)
                else:
                    concat_output = tf.concat(self.attention_output_list, axis=2)
                dense_output = self.dense(concat_output)
                self.theta = tf.reshape(dense_output, (tf.shape(dense_output)[0], self.num_parts, 3, 4))
                return self.theta

            else:
                if decomposer_output is None:
                    decomposer_output = self.decomposer(inputs, training=training)
                out1, out2, out3, out4, decoder_outputs = [], [], [], [], []
                decoder_inputs = tf.transpose(decomposer_output, (1, 0, 2))
                for each in decoder_inputs:
                    decoder_output = self.part_decoder(each, training=training)
                    out1.append(self.part_decoder.out1)
                    out2.append(self.part_decoder.out2)
                    out3.append(self.part_decoder.out3)
                    out4.append(self.part_decoder.out4)
                    decoder_outputs.append(decoder_output)
                # stacked_decoded_parts should be in the shape of (B, num_parts, H, W, D, C)
                inter1 = tf.stack(out1, axis=1)
                inter2 = tf.stack(out2, axis=1)
                inter3 = tf.stack(out3, axis=1)
                inter4 = tf.stack(out4, axis=1)
                self.stacked_decoded_parts = tf.stack(decoder_outputs, axis=1)
                self.attention_output_list = list()
                for i, each_attention in enumerate(self.attention_layer_list):
                    if self.which_layer[i] is '0':
                        self.attention_output_list.append(each_attention(decomposer_output, training=training))
                    elif self.which_layer[i] is '1':
                        self.attention_output_list.append(each_attention(inter1, training=training))
                    elif self.which_layer[i] is '2':
                        self.attention_output_list.append(each_attention(inter2, training=training))
                    elif self.which_layer[i] is '3':
                        self.attention_output_list.append(each_attention(inter3, training=training))
                    elif self.which_layer[i] is '4':
                        self.attention_output_list.append(each_attention(inter4, training=training))
                    elif self.which_layer[i] is '5':
                        self.attention_output_list.append(each_attention(self.stacked_decoded_parts, training=training))
                    else:
                        raise ValueError('which_layer should be one or more of 0, 1, 2, 3, 4 and 5')
                if self.keep_channel:
                    self.temp = list()
                    for each, conv in zip(self.attention_output_list, self.conv_list):
                        each = tf.transpose(each, (0, 2, 3, 1))  # channel is at last dimension
                        each = tf.reshape(conv(each), (each.shape[0], each.shape[1], -1))  # 1D convolution
                        self.temp.append(each)
                    concat_output = tf.concat(self.temp, axis=2)
                else:
                    concat_output = tf.concat(self.attention_output_list, axis=2)
                dense_output = self.dense(concat_output)
                self.theta = tf.reshape(dense_output, (tf.shape(dense_output)[0], self.num_parts, 3, 4))
                resampling_inputs = (self.stacked_decoded_parts, self.theta)
                stacked_transformed_parts = self.resampling(resampling_inputs)
                return stacked_transformed_parts

        else:
            if self.training_process == 1 or self.training_process == '1':
                # decomposer output has shape (B, num_parts, encoding_dims)
                if decomposer_output is None:
                    decomposer_output = self.decomposer(inputs, training=training)
                decoder_outputs = list()
                decoder_inputs = tf.transpose(decomposer_output, (1, 0, 2))
                for each in decoder_inputs:
                    decoder_outputs.append(self.part_decoder(each, training=training))
                # stacked_decoded_parts should be in the shape of (B, num_parts, H, W, D, C)
                self.stacked_decoded_parts = tf.stack(decoder_outputs, axis=1)
                return self.stacked_decoded_parts

            elif self.training_process == 2 or self.training_process == '2':
                # decomposer output has shape (B, num_parts, encoding_dims)
                if decomposer_output is None:
                    decomposer_output = self.decomposer(inputs, training=training)
                decoder_outputs = list()
                decoder_inputs = tf.transpose(decomposer_output, (1, 0, 2))
                for each in decoder_inputs:
                    decoder_outputs.append(self.part_decoder(each, training=training))
                # stacked_decoded_parts should be in the shape of (B, num_parts, H, W, D, C)
                self.stacked_decoded_parts = tf.stack(decoder_outputs, axis=1)
                # summed_inputs should be in the shape of (B, encoding_dims)
                summed_inputs = tf.reduce_sum(decomposer_output, axis=1)
                localization_inputs = (self.stacked_decoded_parts, summed_inputs)
                self.theta = self.localization_net(localization_inputs, training=training)
                return self.theta

            else:
                # decomposer output has shape (B, num_parts, encoding_dims)
                if decomposer_output is None:
                    decomposer_output = self.decomposer(inputs, training=training)
                decoder_outputs = list()
                decoder_inputs = tf.transpose(decomposer_output, (1, 0, 2))
                for each in decoder_inputs:
                    decoder_outputs.append(self.part_decoder(each, training=training))
                # stacked_decoded_parts should be in the shape of (B, num_parts, H, W, D, C)
                self.stacked_decoded_parts = tf.stack(decoder_outputs, axis=1)
                # summed_inputs should be in the shape of (B, encoding_dims)
                summed_inputs = tf.reduce_sum(decomposer_output, axis=1)
                localization_inputs = (self.stacked_decoded_parts, summed_inputs)
                self.theta = self.localization_net(localization_inputs, training=training)
                resampling_inputs = (self.stacked_decoded_parts, self.theta)
                stacked_transformed_parts = self.resampling(resampling_inputs)
                return stacked_transformed_parts

    def train_step(self, data):

        # x has shape (B, H, W, D, C), label has shape (B, num_parts, H, W, D, C), trans has shape (B, num_parts, 3, 4)
        x, label, trans = data

        if self.training_process == 1 or self.training_process == '1':  # training process for pretraining BinaryShapeEncoder, Projection, PartDecoder
            with tf.GradientTape() as tape:
                stacked_decoded_parts = self(x, training=True)
                pi_loss = self._cal_pi_loss()
                part_recon_loss = self._cal_part_reconstruction_loss(label, stacked_decoded_parts)
                total_loss = pi_loss + part_recon_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.pi_loss_tracker.update_state(pi_loss)
            self.part_reconstruction_loss_tracker.update_state(part_recon_loss)
            self.total_loss_tracker.update_state(total_loss)

            return {'PI_Loss': self.pi_loss_tracker.result(),
                    'Part_Recon_Loss': self.part_reconstruction_loss_tracker.result(),
                    'Total_Loss': self.total_loss_tracker.result()}

        elif self.training_process == 2 or self.training_process == '2':
            if self.use_attention:
                weights_list = list()
                for layer in self.attention_layer_list:
                    weights_list.extend(layer.trainable_weights)
                weights_list.extend(self.dense.trainable_weights)
                if self.keep_channel:
                    for each_conv in self.conv_list:
                        weights_list.extend(each_conv.trainable_weights)
                with tf.GradientTape() as tape:
                    theta = self(x, training=True)
                    trans_loss = self._cal_transformation_loss(trans, theta)
                    if self.use_ac_loss:
                        if len(self.attention_output_list) < 2:
                            raise ValueError('which_layer should at least contain 2 layers when use_ac_loss is True!')
                        ac_loss = self._cal_ac_loss(self.temp) if self.keep_channel else self._cal_ac_loss(self.attention_output_list)
                        total_loss = trans_loss + ac_loss
                if self.use_ac_loss:
                    grads = tape.gradient(total_loss, weights_list)
                    self.optimizer.apply_gradients(zip(grads, weights_list))
                    self.transformation_loss_tracker.update_state(trans_loss)
                    self.ac_loss_tracker.update_state(ac_loss)
                    self.total_loss_tracker.update_state(total_loss)
                    return {'Transformation_Loss': self.transformation_loss_tracker.result(),
                            'AC_Loss': self.ac_loss_tracker.result(),
                            'Total_Loss': self.total_loss_tracker.result()}
                else:
                    grads = tape.gradient(trans_loss, weights_list)
                    self.optimizer.apply_gradients(zip(grads, weights_list))
                    self.transformation_loss_tracker.update_state(trans_loss)
                    return {'Transformation_Loss': self.transformation_loss_tracker.result()}
            else:
                with tf.GradientTape() as tape:
                    theta = self(x, training=True)
                    trans_loss = self._cal_transformation_loss(trans, theta)
                grads = tape.gradient(trans_loss, self.localization_net.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.localization_net.trainable_weights))
                self.transformation_loss_tracker.update_state(trans_loss)
                return {'Transformation_Loss': self.transformation_loss_tracker.result()}

        elif self.training_process == 3 or self.training_process == '3':
            with tf.GradientTape() as tape:
                stacked_transformed_parts = self(x, training=True)
                pi_loss = self._cal_pi_loss()
                part_recon_loss = self._cal_part_reconstruction_loss(label, self.stacked_decoded_parts)
                trans_loss = self._cal_transformation_loss(trans, self.theta)
                shape_recon_loss = self._cal_shape_reconstruction_loss(x, tf.reduce_max(stacked_transformed_parts, axis=1))
                if self.use_ac_loss:
                    if len(self.attention_output_list) < 2:
                        raise ValueError('which_layer should at least contain 2 layers when use_ac_loss is True!')
                    ac_loss = self._cal_ac_loss(self.temp) if self.keep_channel else self._cal_ac_loss(self.attention_output_list)
                    total_loss = pi_loss + part_recon_loss + 10 * trans_loss + ac_loss + 10 * shape_recon_loss
                else:
                    total_loss = pi_loss + part_recon_loss + 10 * trans_loss + 10 * shape_recon_loss
            if self.use_ac_loss:
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.pi_loss_tracker.update_state(pi_loss)
                self.part_reconstruction_loss_tracker.update_state(part_recon_loss)
                self.transformation_loss_tracker.update_state(trans_loss)
                self.ac_loss_tracker.update_state(ac_loss)
                self.shape_reconstruction_loss_tracker.update_state(shape_recon_loss)
                self.total_loss_tracker.update_state(total_loss)
                return {'PI_Loss': self.pi_loss_tracker.result(),
                        'Part_Recon_Loss': self.part_reconstruction_loss_tracker.result(),
                        'Transformation_Loss': self.transformation_loss_tracker.result(),
                        'AC_Loss': self.ac_loss_tracker.result(),
                        'Shape_Recon_Loss': self.shape_reconstruction_loss_tracker.result(),
                        'Total_Loss': self.total_loss_tracker.result()}
            else:
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.pi_loss_tracker.update_state(pi_loss)
                self.part_reconstruction_loss_tracker.update_state(part_recon_loss)
                self.transformation_loss_tracker.update_state(trans_loss)
                self.shape_reconstruction_loss_tracker.update_state(shape_recon_loss)
                self.total_loss_tracker.update_state(total_loss)
                return {'PI_Loss': self.pi_loss_tracker.result(),
                        'Part_Recon_Loss': self.part_reconstruction_loss_tracker.result(),
                        'Transformation_Loss': self.transformation_loss_tracker.result(),
                        'Shape_Recon_Loss': self.shape_reconstruction_loss_tracker.result(),
                        'Total_Loss': self.total_loss_tracker.result()}

        else:
            raise ValueError('training process should be one of 1, 2 and 3')

    def _cal_pi_loss(self):
        params = list()
        for each_layer in self.decomposer.projection_layer_list:
            params.append(each_layer.trainable_weights[0])
        # params should be list of tensor, whose elements are the trainable weights of Projection layer
        params_tensor = tf.convert_to_tensor(params)
        # params_tensor has shape (num_parts, encoding_dims, encoding_dims)
        pi_loss1 = tf.reduce_sum(tf.norm(params_tensor ** 2 - params_tensor, axis=[-2, -1]) ** 2)
        pi_loss2 = list()
        for idx, each_param in enumerate(params_tensor):
            unstack_params = tf.unstack(params_tensor, axis=0)
            del unstack_params[idx]
            new_params = tf.convert_to_tensor(unstack_params)
            pi_loss2.append(tf.norm(each_param * new_params, axis=[-2, -1]) ** 2)
        pi_loss2 = tf.reduce_sum(pi_loss2)
        pi_loss3 = tf.norm(tf.reduce_sum(params_tensor, axis=0) - tf.eye(tf.shape(params_tensor)[1])) ** 2
        return pi_loss1 + pi_loss2 + pi_loss3

    # @staticmethod
    # def _cal_part_reconstruction_loss(gt, pred):
    #     return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt, pred), axis=(1, 2, 3, 4)))

    def _cal_part_reconstruction_loss(self, gt, pred):
        pred = tf.clip_by_value(pred, 1e-7, 1.-1e-7)
        bce = 2 * tf.reduce_sum(-self.bce_weight * gt * tf.math.log(pred + 1e-7) - (1 - self.bce_weight) * (1 - gt) * tf.math.log(1 - pred + 1e-7), axis=(1, 2, 3, 4, 5))
        return tf.reduce_mean(bce)

    # @staticmethod
    # def _cal_shape_reconstruction_loss(gt, pred):
    #     return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt, pred), axis=(1, 2, 3)))

    def _cal_shape_reconstruction_loss(self, gt, pred):
        pred = tf.clip_by_value(pred, 1e-7, 1.-1e-7)
        bce = 2 * tf.reduce_sum(-self.bce_weight_shape * gt * tf.math.log(pred + 1e-7) - (1 - self.bce_weight_shape) * (1 - gt) * tf.math.log(1 - pred + 1e-7), axis=(1, 2, 3, 4))
        return tf.reduce_mean(bce)

    @staticmethod
    def _cal_transformation_loss(gt, pred):
        return tf.nn.l2_loss(gt-pred) / tf.cast(tf.shape(gt)[0], dtype=tf.float32)

    @staticmethod
    def _cal_ac_loss(pred):
        loss = list()
        for i in range(len(pred)):
            for j in range(len(pred)):
                if i < j:
                    loss.append(tf.math.square(pred[i] - pred[j]))
                else:
                    continue
        return tf.reduce_sum(loss) / len(loss)

    @property
    def metrics(self):
        return [self.pi_loss_tracker, self.part_reconstruction_loss_tracker, self.transformation_loss_tracker,
                self.ac_loss_tracker, self.shape_reconstruction_loss_tracker, self.total_loss_tracker,
                self.transformation_mse_tracker, self.all_part_mIoU_tracker, self.shape_mIoU_tracker] + self.part_mIoU_tracker_list

    def test_step(self, data):
        x, labels, trans = data
        if self.training_process == 1 or self.training_process == '1':
            parts = self(x, training=False)
            parts = tf.transpose(tf.where(parts > 0.5, 1., 0.), (1, 0, 2, 3, 4, 5))
            labels = tf.transpose(labels, (1, 0, 2, 3, 4, 5))
            for gt, part, part_mIoU_tracker in zip(labels, parts, self.part_mIoU_tracker_list):
                part_mIoU_tracker.update_state(gt, part)
                self.all_part_mIoU_tracker.update_state(gt, part)
            metrics_dict = {f'Part{i+1}_mIoU': self.part_mIoU_tracker_list[i].result() for i in range(self.num_parts)}
            dict1 = {'Part_mIoU': self.all_part_mIoU_tracker.result()}
            metrics_dict.update(dict1)
            return metrics_dict

        elif self.training_process == 2 or self.training_process == '2':
            theta = self(x, training=False)
            trans_mse = self._cal_transformation_loss(trans, theta) * 2 / self.num_parts
            self.transformation_mse_tracker.update_state(trans_mse)
            shapes = Resampling()((self.stacked_decoded_parts, theta))
            shapes = tf.where(tf.reduce_max(shapes, axis=1) > 0.5, 1., 0.)
            self.shape_mIoU_tracker.update_state(x, shapes)
            return {'Transformation_MSE': self.transformation_mse_tracker.result(),
                    'Shape_mIoU': self.shape_mIoU_tracker.result()}

        else:
            shapes = self(x, training=False)
            trans_mse = self._cal_transformation_loss(trans, self.theta) * 2 / self.num_parts
            self.transformation_mse_tracker.update_state(trans_mse)
            shapes = tf.where(tf.reduce_max(shapes, axis=1) > 0.5, 1., 0.)
            parts = tf.transpose(tf.where(self.stacked_decoded_parts > 0.5, 1., 0.), (1, 0, 2, 3, 4, 5))
            labels = tf.transpose(labels, (1, 0, 2, 3, 4, 5))
            for gt, part, part_mIoU_tracker in zip(labels, parts, self.part_mIoU_tracker_list):
                part_mIoU_tracker.update_state(gt, part)
                self.all_part_mIoU_tracker(gt, part)
            self.shape_mIoU_tracker.update_state(x, shapes)
            metrics_dict = {f'Part{i+1}_mIoU': self.part_mIoU_tracker_list[i].result() for i in range(self.num_parts)}
            dict1 = {'Part_mIoU': self.all_part_mIoU_tracker.result(), 'Transformation_MSE': self.transformation_mse_tracker.result(),
                     'Shape_mIoU': self.shape_mIoU_tracker.result()}
            metrics_dict.update(dict1)
            return metrics_dict

