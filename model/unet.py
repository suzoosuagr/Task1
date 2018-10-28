import logging
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from model.layer_utils import conv2d
from model.layer_utils import common_layer
from model.layer_utils import dropout
from model.layer_utils import maxpool
from model.layer_utils import deconv2d
from model.layer_utils import crop_and_concat
# from model.layer_utils import pixel_wise_softmax


def Build_Unet(is_training, inputs, params):
    """Build the Unet graph
    

    Args:
        is_training: 'bool' type, indicate the batch normalization 
        inputs: [batch, height, weight, channels]
        params: (dict) contains the params from params.json

    Return: 
        logits: result logits of the Unet. 
    """
    # logging the basic parameters of Unet
    logging.info(
        "Layers: {num_layers}, channels_root: {num_channels}, image_size: {image_size}, num_epochs: {num_epochs}\n".format(
            num_layers=params.num_layers,
            num_channels=params.num_channels,
            image_size=params.image_size,
            num_epochs=params.num_epochs
        )
    )

    logging.info(
        "Loss_function: {loss_function}, Regularization:{regularization}, Optimizer: {optimizer}, Net_type: {Net_type}".format(
            loss_function=params.cost_type,
            optimizer=params.optimizer,
            Net_type=params.Net_type,
            regularization=params.regularizer
        )
    )
    logging.critical("Using Thres={threshold} -> pred\nUsing SoftDice\n")


    output = inputs['images']

    # ----------------------------Net Architecture Build ------------------------------- #
    
    dw_convs = OrderedDict()
    up_convs = OrderedDict()
    pools = OrderedDict()
    Laplacian = OrderedDict()
    # Down-Conv
    for layer in range(params.num_layers):
        if layer == 0:
            num_channels = 16
        else:
            num_channels = layer * 2 * params.num_channels

        with tf.variable_scope("Down_conv_{}".format(layer + 1)):          
            # down_conv
            output = conv2d(output, params.filter_height, params.filter_width, num_channels, params.stride_y, params.stride_x, "conv2d_c_extend", "SAME")
            output = dropout(output, params.keep_prob)
            output = common_layer(output, params.bn_momentum, is_training)
            # same_size_conv
            output = conv2d(output, params.filter_height, params.filter_width, num_channels, params.stride_y, params.stride_x, "conv2d_c_keep", "SAME")
            output = dropout(output, params.keep_prob)
            output = common_layer(output, params.bn_momentum, is_training)
            # put output in ordered Dict
            dw_convs[layer] = output
            if layer < params.num_layers - 1:
                output = maxpool(output, 2,2,2,2, "maxpool")
    
    output = dw_convs[params.num_layers - 1]

    # Up-Conv
    for layer in range(params.num_layers - 2, -1, -1):
        if layer == 0:
            num_channels = 16
        else:
            num_channels = layer * 2 * params.num_channels

        with tf.variable_scope("Up_deconv_{}".format(layer + 1)):
            output = deconv2d(output, 3, 3, num_channels, 2, 2, "transpose_conv", padding='SAME')
            output = dropout(output, params.keep_prob)
            output = common_layer(output, params.bn_momentum, None)
            # do the concat op
            output = crop_and_concat(dw_convs[layer], output)

            output = conv2d(output, 3, 3, num_channels, 1, 1, "conv2d_c_shrink", padding='SAME')
            output = dropout(output, params.keep_prob)
            output = common_layer(output, params.bn_momentum, is_training)
            output = conv2d(output, 3, 3, num_channels, 1, 1, "conv2d_c_keep", padding='SAME')
            output = dropout(output, params.keep_prob)
            output = common_layer(output, params.bn_momentum, is_training)
            # put output(up) to up_conv Dict
            up_convs[layer] = output
    # output map
    conv = conv2d(output, 1, 1, 1, 1, 1, name='1x1_conv', padding='SAME')
    logits = tf.nn.relu(conv)
    
    return logits

def _get_cost(logits, masks, params, class_weights, _variables):
    """Construct the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient
    Args:
        logits: logits map of unet
        masks:  ground truth of the segmentation results
        params: {Dict} contains the params of model

    Optional args(in cost_kwargs)
        class_weights: weights for the different classes in case of multi-class inbalance
    Return:
        loss: loss function
    """
    with tf.name_scope("cost"):
        flat_logits = tf.reshape(logits, [-1,params.num_labels])
        flat_masks = tf.reshape(masks, [-1, params.num_labels])
        if params.cost_type == "cross_entropy":
            if class_weights:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
                weight_map = tf.multiply(flat_masks, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_masks)

                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_masks))

        # using dice_coefficient is better
        elif params.cost_type == "dice_coefficient":
            eps = 1e-5
            num = tf.cast(tf.shape(logits)[0], tf.float32)
            probs = tf.nn.sigmoid(logits)
            intersection = tf.reduce_sum(probs * masks)
            union = eps + tf.reduce_sum(probs) + tf.reduce_sum(masks)
            dice_coeff = -(2 * intersection / union)
            dice_coeff = 1 - dice_coeff / num

            flat_probs = tf.reshape(probs, [-1])
            flat_masks = tf.reshape(masks, [-1])
            bce_loss = tf.keras.backend.binary_crossentropy(flat_masks, flat_probs)

            loss = tf.reduce_sum(dice_coeff + bce_loss)


        else:
            raise ValueError('Unknown cost type {}'.format(params.cost_type))
        
        if params.regularizer:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in _variables])
            loss += (params.regularizer * regularizers)
        
        return loss

def _get_optimizer(loss, params):
    global_step = tf.train.get_or_create_global_step()
    if params.optimizer == 'momentum':
        learning_rate = params.learning_rate_f_momentum
        decay_rate = params.decay_rate
        momentum = params.momentum
        decay_steps = params.decay_steps

        learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node,
                                                momentum=momentum,).minimize(loss=loss, global_step=global_step)
    elif params.optimizer == 'adam':
        learning_rate = params.learning_rate
        decay_rate = params.decay_rate
        decay_steps = params.decay_steps
        learning_rate_node = tf.Variable(learning_rate, name='learning_rate')
        learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node).minimize(loss=loss, global_step=global_step)
    
    else:
        raise ValueError("Unknown optimizer type:{}, choose from adam and momentum".format(params.optimizer))

    return optimizer, learning_rate_node







