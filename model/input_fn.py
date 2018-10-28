import tensorflow as tf
import numpy as np
# from model.utils import show_tensor_as_image

def _parse_function(filename, mask_filename, size):
    """Obtain the image from the filename (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
        - Decode the mask from png format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)
    mask_string = tf.read_file(mask_filename)
g
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    mask_decoded = tf.image.decode_png(mask_string, channels=1)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # NOTE: the per image standardization is added to test pre-programming.  
    image = tf.image.per_image_standardization(image)
    mask = tf.image.convert_image_dtype(mask_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, size)
    resized_mask = tf.image.resize_images(mask, size)

    return resized_image, resized_mask


def train_preprocess(image, mask, use_random_flip):
    """Image preprocessing for training.
    Apply the following operations:
        - random brightness augment for image
        - concat image and mask together to process flop and rotation ops
        - random flip (mix)
        - random rotation (mix)
        - split mix block to image and mask

    Args:
        image:  tensor '[weight, height, 3]' dtype = 'tf.float32' 
        mask:   tensor '[weight, height, 1]' dtype = 'tf.float32'
    
    Return:
        image:  tensor '[weight, height, 3]' of type 'tf.float32' 
        mask:   tensor '[weight, height, 1]' of type 'tf.float32'
    """
    # shot_cut = image

    image = tf.image.random_brightness(image, max_delta = 32.0 / 255.0)

    mix = tf.concat([image, mask],2)

    if use_random_flip:
        mix = tf.image.random_flip_left_right(mix)
        mix = tf.image.random_flip_up_down(mix)

    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
 
    shape = tf.shape(mix)
    times = np.random.randint(0,4)
    mix = tf.image.rot90(mix, k=times)
    mix = tf.reshape(mix,shape)
    image, mask = tf.split(mix, [3,1], axis = 2)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    mask = tf.cast(mask > 0.5, tf.float32)
    # mask = tf.clip_by_value(mask, 0.0, 1.0)
    """
    NOTE: When I turn on clip mask function, the augmented mask shows strips. In my mind, this
    is because the clip function changed value. For ground truth, we'd better keep it's value.
    Don't use progress which may change the value of mask. We need use a threshold to literally
    split mask value to 0.0 and 1.0
    """

    return image, mask


def input_fn(is_training, filenames, masks, params):
    """Input function for the PH2Ddataset.
    The filenames have format "{label}_IMG_{id}.bmp".
    For instance: "data_dir/2_IMG_4584.bmp".
    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.bmp"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(masks), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(masks)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:

        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(masks)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, masks = iterator.get_next()
    laplacian = _get_laplacian(images)
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'masks': masks, 'iterator_init_op': iterator_init_op, "laplacian": laplacian}
    # show_tensor_as_image(inputs)
    return inputs


def _get_laplacian(images):
    g_images = tf.image.rgb_to_grayscale(images)
    l_filter = tf.constant([[0,1,0], [1,-4,1], [0,1,0]], tf.float32)
    l_filter = tf.reshape(l_filter, [3,3,1,1])
    laplacian = tf.nn.conv2d(g_images, l_filter, strides=[1,1,1,1], padding='SAME')
    return laplacian
