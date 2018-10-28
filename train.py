import argparse
import random
import logging
import os 
import csv
import numpy as np

import tensorflow as tf

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default = 'experiments/test', 
                    help = "loading params.json")
parser.add_argument('--data_dir', default = '../data/Training/ISIC2018_Task1_Training_shrink',
                    help = "Dictory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help = "Optional, directory or file containing weights to reload before training")
parser.add_argument('--memo', default = None,
                    help = "write some note into log file")
parser.add_argument('--ground_truth_dir', default='data/Training/ISIC2018_Task1_Training_shrink_GroundTruth', 
                    help = "Directory of GT masks")
parser.add_argument('--label_dir', default = 'data/Training/ground_truth.csv',
                    help = "Directory of label csv file")


def get_label(csv_dir = None, Traindata_dir = None, Devdata_dir = None):
    """To get the labels of train and eval images respectively 
    
    The following operations applied
    - read csv data
    - create dict indicate imgs_id with onehot labels
    - read imgs_id in folder 
    - return train_labels and eval_labels with the same ranks of filenames in folders

    --- TODO: using argmax to transform the one hot to normal
    
    by JWang
    """
    with open(csv_dir) as f:
        reader = csv.reader(f)
        next(reader)
        # data = [r for r in reader]
        data = list(reader)
    data = np.array(data)
    keys = [str(row) for row in data[:,0]]
    labels = data[:,1::].astype(float)
    labels = np.argmax(labels,axis=1)

    dictx = {}
    for i in range(len(keys)):
        dictx[keys[i]] = labels[i]

    train_images_id = [f.split('.')[0] for f in os.listdir(Traindata_dir) if f.endswith('.jpg')]
    eval_images_id = [f.split('.')[0] for f in os.listdir(Devdata_dir) if f.endswith('.jpg')]
    train_labels = []
    eval_labels = []

    print(dictx['ISIC_0000008'])

    for i in train_images_id:
        train_labels.append(dictx[i])
    for i in eval_images_id:
        eval_labels.append(dictx[i])

    # train_labels = np.array(train_labels).
    # eval_labels = np.array(eval_labels)

    return train_labels, eval_labels

# def get_mask()

if __name__ == '__main__':
    args = parser.parse_args()
    tf.set_random_seed(233)

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "could't find the json configuration file at {}".format(json_path)
    params = Params(json_path)

    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir,"best_weight"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    ground_truth_dir = args.ground_truth_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")
    train_masks_dir = os.path.join(ground_truth_dir, "train")
    dev_masks_dir = os.path.join(ground_truth_dir, "dev")

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg')]
    eval_filenames  = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                       if f.endswith('.jpg')]

    # Get the filenames from the train and dev sets of masks 
    train_masks_filenames = [os.path.join(train_masks_dir, f) for f in os.listdir(train_masks_dir) if f.endswith('.png')]
    eval_masks_filenames = [os.path.join(dev_masks_dir, f) for f in os.listdir(dev_masks_dir) if f.endswith('.png')]

    # Get the images id
    # assert os.path.isfile(args.label_dir), "Could't find the label file in {} ".format(args.label_dir)

    # train_labels, eval_labels = get_label(args.label_dir, train_data_dir, dev_data_dir)

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    # PH2Dataset 0.25 scale,size 191, 143
    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_masks_filenames, params)
    eval_inputs = input_fn(False, eval_filenames, eval_masks_filenames, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
    