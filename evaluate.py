'''
evaluate.py

Based on https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/
'''

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import util
import model.began as began
import data.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/CelebA', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/began_base', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(g, d, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.util.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    g.eval()
    d.eval()

    # summary for current eval loop
    summ = []

    z_fixed = torch.FloatTensor(params.batch_size, params.h).uniform_(-1,1).to(params.device)

    # compute metrics over the dataset
    for batch_data in dataloader:
        r_img = batch_data[0]

        # move to GPU if available
        if params.cuda: r_img = r_img.cuda(async=True)
        # fetch the next evaluation batch
        #data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        g_img = g(z_fixed)

        g_img_passed = d(g_img)
        r_img_passed = d(r_img)

        g_loss = g.loss_fn(g_img, g_img_passed)
        d_loss = d.loss_fn(r_img, g_img, r_img_passed, g_img_passed)
        b_converge = began_convergence(r_img, g_img, r_img_passed, g_img_passed, params.began_gamma)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        # output_batch = output_batch.data.cpu().numpy()
        # labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](r_img, g_img, r_img_passed, g_img_passed)
                          for metric in metrics}
        summary_batch['g_loss'] = g_loss.data
        summary_batch['d_loss'] = d_loss.data
        summary_batch['b_converge'] = b_converge.data
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


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

    # Get the logger
    util.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    test_dl = data_loader.fetch_dataloader('data', 'test', params, shuffle=False)

    logging.info("- done.")

    # Define the model
    g = began.BeganGenerator(params).to(params.device)
    d = began.BeganDiscriminator(params).to(params.device)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    util.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), g, d)

    # Evaluate
    test_metrics = evaluate(g, d, test_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    util.save_dict_to_json(test_metrics, save_path)
