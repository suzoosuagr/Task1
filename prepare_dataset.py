import argparse
import random
import os
import csv
import numpy as np

from PIL import Image
from tqdm import tqdm

SIZE = (300,226)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = 'data/data_ground_truth', help = "Directory with Task1 data")
parser.add_argument('--output_dir', default = 'data/ground_truth', help = "where to write the new data")
parser.add_argument('--format', default = '.png', choices = ['.jpg', '.png'], help = "select the image format from .jpg & .png")

def resize_save(filename, output_dir, resample, size = SIZE):
    image = Image.open(filename)
    # need to use bilinear
    image = image.resize(size, resample)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))

if __name__ == '__main__':

    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    if args.format == '.png':
        resample = Image.NEAREST
    else:
        resample = Image.BILINEAR
    
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith(args.format)]
    
    # Define the data directiories
    random.seed(233)
    filenames.sort()
    random.shuffle(filenames)

    split_1 = int(0.8*len(filenames))
    split_2 = int(0.9*len(filenames))

    train_filenames = filenames[:split_1]
    dev_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]

    filenames = {'train': train_filenames,
                 'dev':   dev_filenames,
                 'test':  test_filenames}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print("warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir  {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split,output_dir_split))
        for img in tqdm(filenames[split]):
            resize_save(img, output_dir_split,resample, size = SIZE)

    print("Done building dataset")

