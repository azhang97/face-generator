'''
test.py

Based on https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/
'''

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import util
import model.began as began
import data.data_loader as data_loader
import torchvision.utils as torch_utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/began_base', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--num', default=1, type=int, help='Number of images to create')


def test(z, g, d):
    """Test the model on `num_steps` batches.
    Args:
        z
        g
        d
    """

    # set model to evaluation mode
    g.eval()
    d.eval()

    return g(z)


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = util.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    if params.ngpu > 0 and params.cuda: params.device = torch.device('cuda')
    else: params.device = torch.device('cpu')

    # Set the random seed for reproducible experiments
    torch.manual_seed(42)
    if params.cuda: torch.cuda.manual_seed(42)

    # Define the model
    g = began.BeganGenerator(params).to(params.device)
    d = began.BeganDiscriminator(params).to(params.device)


    # Reload weights from the saved file
    util.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), g, d)

    # create fixed z vector
    z_fixed = torch.FloatTensor(args.num, params.h).uniform_(-1,1).to(params.device)

    # test
    f_img = test(z_fixed, g, d)

    save_path = os.path.join(args.model_dir, "test.jpg")
    torch_utils.save_image(f_img, save_path)
