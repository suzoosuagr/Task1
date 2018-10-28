import logging
import json
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf

class Params:
    """
    read parameters from json file

    Example:

    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        self.update(json_path)

    def save(self,json_path):
        with open(json_path,'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def update(self,json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

# def show_tensor_as_image(inputs):
#     image = inputs['laplacian']
#     # mask = inputs['masks']
#     # image_shape = inputs['images'].get_shape().as_list()
#     # mask_shape = inputs['masks'].get_shape().as_list()

#     init = tf.global_variables_initializer()
#     # init_2 = tf.initialize_all_variables()

#     with tf.Session() as sess:
#         sess.run(init)
#         # sess.run(init_2)
#         sess.run(inputs['iterator_init_op'])

#         # sess.run(print(image_shape, mask_shape))
#         images = image.eval()
#         # masks = mask.eval()

#         fig = plt.figure()
#         plt.gray()
#         columns = 4
#         rows = 2
#         for i in range(1, columns*rows+1):
#             img = images[i-1]
#             img = np.reshape(img, [300, 226])
#             fig.add_subplot(rows,columns,i)
#             plt.imshow(img)
#         plt.show()
#         print('showed')




