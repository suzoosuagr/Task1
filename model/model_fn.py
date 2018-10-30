import tensorflow as tf
from model.unet import Build_Unet
from model.unet import _get_cost
from model.unet import _get_optimizer

from model.L_Unet import Build_L_Unet
from model.L_Unet import _get_cost_L
from model.L_Unet import _get_optimizer_L

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.
    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights
    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    masks = inputs['masks']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        # logits = AlexNet(is_training, inputs, params)
        if params.Net_type == "Unet_ori":
            logits = Build_Unet(is_training, inputs, params)
        elif params.Net_type == "Unet_laplacian":
            logits = Build_L_Unet(is_training, inputs, params)
        else: 
            raise ValueError("unknown net type{}".format(params.Net_type))
        class_weight = {}
        # using pixel wise softmax to get the result. 
        _Variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')

        if params.Net_type == "Unet_ort":
            loss = _get_cost(logits, masks, params, class_weight, _Variables)
        else:
            loss = _get_cost_L(logits, masks, params, class_weight, _Variables)
        # gradients_node = tf.gradients(loss, _Variables)
        probs = tf.nn.sigmoid(logits)
        predictions = tf.cast((probs > params.threshold), tf.float32)


    # Define loss and accuracy
    # loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
    correct_pred = tf.equal(predictions,masks)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        if params.Net_type == "Unet_ori":
            optimizer, decay_learning_rate = _get_optimizer(loss, params)
        else:
            optimizer, decay_learning_rate = _get_optimizer_L(loss, params)
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer
        else:
            train_op = optimizer


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=masks, predictions=predictions),
            'loss': tf.metrics.mean(loss),
            'IoU':tf.metrics.mean_iou(labels=masks, predictions=predictions, num_classes=2)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    """
    TODO: uncomment this block to unlock the function to check how fail happens
    """
    # Add incorrectly labeled images
    # fails = tf.not_equal(tf.argmax(predictions,3), tf.argmax(masks, 3))

    # # Add a different summary to know how they were misclassified
    # for label in range(0, params.num_labels):
    #     fails_label = tf.logical_and(fails, correct_pred)
    #     incorrect_image_label = tf.boolean_mask(inputs['images'], fails_label)
    #     tf.summary.image('incorrectly_labeled_{}', incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
