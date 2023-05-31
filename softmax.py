from tensorflow import keras
from tensorflow.keras import backend as K

from Scaled_DotProductSoftAttention import ScaledDotProductSoftAttention


class SoftmaxLayer(keras.layers.Layer):
    """Multi-head attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,          
                 **kwargs):
        """Initialize the layer.
    
        """
        self.supports_masking = True
        

        self.intensity = self.attention = None
        super(SoftmaxLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            
        }
        base_config = super(SoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k = input_shape
            return q[:-1] + (k[-2],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k= input_shape
        else:
            q = k = input_shape
        feature_dim = int(k[-1])
        
        super(SoftmaxLayer, self).build(input_shape)
    """
    @staticmethod

    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        return K.permute_dimensions(x, [0, 2, 1, 3])

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))
    
    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))
    """
    def call(self, inputs, mask=None):
        print("inputs:",inputs)
        print("mask:",mask)
        if isinstance(inputs, list):
            q, k = inputs
        else:
            q = k = inputs
        print("q.shape:",q.shape)
        print("k.shape:",k.shape)
        if isinstance(mask, list):
            q_mask, k_mask = mask
        else:
            q_mask = k_mask = mask
        print("q_mask.shape:",q_mask.shape)
        print("k_mask.shape:",k_mask.shape)
        scaled_dot_product_attention = ScaledDotProductSoftAttention(
            name='%s-Attention' % self.name,
        )
        y = scaled_dot_product_attention(
            inputs=[
                q,
                k,
              
            ],
            mask=[
                q_mask,
                k_mask,
            ],
        )
        """
        self.intensity = self._reshape_attention_from_batches(scaled_dot_product_attention.intensity, self.head_num)
        self.attention = self._reshape_attention_from_batches(y, self.head_num)
        """

        # Add shape information to tensor
        input_shape = [K.int_shape(q), K.int_shape(k)]
        output_shape = self.compute_output_shape(input_shape)
        if output_shape[1] is not None:
            output_shape = (-1,) + output_shape[1:]
            y = K.reshape(y, output_shape)
        return y