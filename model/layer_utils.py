import tensorflow as tf

def common_layer(output, bn_momentum, is_training):
    output = tf.layers.batch_normalization(output, momentum=bn_momentum, training=is_training)
    output = tf.nn.relu(output)
    return output

def conv2d(output, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding = 'SAME'):
    input_channels = int(output.get_shape()[-1])
    with tf.name_scope(name):    
        with tf.variable_scope(name + 'var'):
            weights = tf.get_variable('weights', shape= [filter_height,
                                                        filter_width,
                                                        input_channels,
                                                        num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
        conv = tf.nn.conv2d(output, weights, strides=[1,stride_y, stride_x,1], padding=padding, name='conv2d')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        return bias

def dropout(output, keep_prob_):
    return tf.nn.dropout(output, keep_prob_)

def maxpool(output, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(output,ksize = [1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],padding=padding,name=name)

def deconv2d(output, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding = 'SAME'):
    """This is conv2d_transpose, for upsampling usage

    Args:
        output: A 4D `Tensor` from previous layer
        filter_height, filter_width, num_filters, stride_y, stride_x: are basic params for conv_transpose ops
        name: Name of op
        padding: `SAME` or `VALID`
    Return:
        A `Tensor` as the output with 4-D geology 
    """
    shape = tf.shape(output)
    static_shape = output.get_shape().as_list()
    
    with tf.name_scope(name):
        output_shape = tf.stack([shape[0], static_shape[1]*2, static_shape[2]*2, num_filters])
        # static_output_shape = tf.stack([static_shape[0], static_shape[1] * 2, static_shape[2] * 2, static_shape[3] // 2])
        # output_shape = [-1, shape[1]*2, shape[2]*2, shape[3]//2]
        with tf.variable_scope(name + 'var'):
            weights = tf.get_variable('weights', shape= [filter_height, 
                                                         filter_width, 
                                                         num_filters,
                                                         static_shape[3]])
            biases = tf.get_variable('biases', shape=[num_filters])
        conv = tf.nn.conv2d_transpose(output, weights, output_shape, strides=[1, stride_y, stride_x, 1], padding=padding, name='conv2d_transpose')
        # conv = tf.reshape(conv, shape = output_shape)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), shape = output_shape)
        return bias
    
def crop_and_concat(x1, x2):
    """ Crop from x1, and concat with x2
        Cropping is applied on x2
    
    Args:
        x1: down_sampled convolution layer tensor
        x2: up_sampling transpose convolution layer
    
    Return:
        a Tensor 
    """
    with tf.name_scope('crop_and_concat'):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left conner of the crop
        offsets = [0,(x2_shape[1] - x1_shape[1])//2, (x2_shape[2] - x1_shape[2])//2, 0]
        size = [-1, x1_shape[1], x1_shape[2], -1]
        x2_crop = tf.slice(x2, offsets, size)
        x2_crop = tf.reshape(x2_crop, shape = x1_shape)
        output = tf.concat([x1, x2_crop],3)
        return output
def crop_and_concat_L(x1, x2):
    with tf.name_scope('crop_and_concat_L'):
        output = tf.concat([x1, x2],3)
        return output

# def concat(x1, x2):
#     with tf.name_scope('Do_concat'):
#         x1_shape = tf.shape()

# TODO: finish pixel_wise_softmax and cross_entropy for loss function and logistic function

def pixel_wise_softmax(output_map):
    """Pixel_wise_softmax used for dice_coefficient
    NOTE: To remain 4 dims output, should let `keepdims = True`. This flag let `reduce_max` and `reduce_sum` remain the operated axis set as 1. 
    Args:
        output_map: The logits of net with dim [batch, height, width, n_classes]
    Return:
        loss: the pixel wise softmax loss, dim [batch, height, width, 1]

        The pixel_wise_softmax is only useful for multi-channel labeled mask, not for one labeled.
    """
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis = 3, keepdims = True)
        exponential_map = tf.exp(output_map)
        normalize = tf.reduce_sum(exponential_map, axis = 3, keepdims = True)
        loss = exponential_map / normalize
        return loss
        