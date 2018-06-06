'''
train.py

Modified from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/train.py
'''
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import util
import model.began as began
import data.data_loader as data_loader
from evaluate import evaluate
import torchvision.utils as torch_utils
from test import test

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/CelebA/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/began_base', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(step, g, d, g_optimizer, d_optimizer, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.util.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    g.train()
    d.train()

    # summary for current training loop and a running average object for loss
    summ = []
    g_loss_avg = util.RunningAverage()
    d_loss_avg = util.RunningAverage()
    b_converge_avg = util.RunningAverage()
    curr_step = step

    z_G = torch.FloatTensor(params.batch_size, params.h).to(params.device)

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, train_batch in enumerate(dataloader):
            r_img = train_batch[0]

            # move to GPU if available
            if params.cuda: r_img = r_img.cuda(async=True)

            # Reset the noise vectors
            z_G.data.uniform_(-1,1)

            ########## Train Discriminator ##########
            g_img = g(z_G)

            g_img_passed = d(g_img.detach())
            r_img_passed = d(r_img)

            d_loss = d.loss_fn(r_img, g_img.detach(), r_img_passed, g_img_passed)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            ########## Train Generator ##########
            g_img_passed = d(g_img)
            g_loss = g.loss_fn(g_img, g_img_passed)


            # clear previous gradients, compute gradients of all variables wrt loss
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()# performs updates using calculated gradients

            # calcualte began convergence measure
            b_converge = began.began_convergence(r_img, g_img, r_img_passed, g_img_passed, params.began_gamma)


            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                # r_img = r_img.data.cpu().numpy()
                # g_img = g_img.data.cpu().numpy()
                # r_img_passed = r_img_passed.cpu().numpy()
                # g_img_passed = g_img_passed.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](g_img, g_img, r_img_passed, g_img_passed) for metric in metrics}
                summary_batch['g_loss'] = g_loss.data
                summary_batch['d_loss'] = d_loss.data
                summary_batch['b_converge'] = b_converge.data
                summ.append(summary_batch)

            # update the average loss
            g_loss_avg.update(g_loss.data)
            d_loss_avg.update(d_loss.data)
            b_converge_avg.update(b_converge.data)

            t.set_postfix(g_loss='{:05.3f}'.format(g_loss_avg()),
                            d_loss='{:05.3f}'.format(d_loss_avg()),
                            converge='{:05.3f}'.format(b_converge_avg()))
            t.update()

            # Apply learning rate update
            g_lr = params.g_learning_rate * params.lr_update ** (curr_step // params.lr_update_step)
            d_lr = params.d_learning_rate * params.lr_update ** (curr_step // params.lr_update_step)
            for pg in g_optimizer.param_groups:
                pg['lr'] = g_lr
            for pg in d_optimizer.param_groups:
                pg['lr'] = d_lr

            curr_step += 1

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return curr_step, b_converge_avg()

def train_and_evaluate(g, d, train_dataloader, val_dataloader, g_optimizer, d_optimizer, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        g
        d
        train_dataloader: (DataLoader) a torch.util.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.util.data.DataLoader object that fetches validation data
        g_optimizer
        d_optimizer
        # metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint = util.load_checkpoint(restore_path, g, d, g_optimizer, d_optimizer)
        d.began_k = checkpoint['began_k']
        g.z_fixed = checkpoint['z_fixed']
        start = checkpoint['epoch']
        step = checkpoint['step']
    else:
        start = 0
        step = 0

    best_b_converge = float('inf')

    # save start images
    if start == 0:
        torch_utils.save_image(test(g.z_fixed, g, d), os.path.join(args.model_dir, 'start.jpg'))

    for epoch in range(start, params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        step, b_converge = train(step, g, d, g_optimizer, d_optimizer, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        #val_metrics = evaluate(g, d, val_dataloader, metrics, params)

        #b_converge = val_metrics['b_converge']
        is_best = b_converge <= best_b_converge
        # Save weights
        if (epoch % params.save_epochs == 0 or epoch == params.num_epochs - 1 or is_best):
            util.save_checkpoint({'epoch': epoch + 1,
                                   'g_state_dict':g.state_dict(),
                                   'd_state_dict':d.state_dict(),
                                   'g_optim_dict':g_optimizer.state_dict(),
                                   'd_optim_dict':d_optimizer.state_dict(),
                                   'began_k':d.began_k,
                                   'z_fixed':g.z_fixed,
                                   'step':step
                                   },
                                   is_best=is_best,
                                   checkpoint=model_dir,
                                   epoch=epoch,
                                   params=params)

        # Save images form this epoch
        torch_utils.save_image(test(g.z_fixed, g, d), os.path.join(args.model_dir, 'epoch{}.jpg'.format(epoch+1)))

        # If best_eval, best_save_path
        # if is_best:
        #    logging.info("- Found new best convergence")
        #    best_b_converge = b_converge

           # Save best val metrics in a json file in the model directory
           # best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
           # util.save_dict_to_json(val_metrics, best_json_path)

        # # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        # util.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = util.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    if params.ngpu > 0 and params.cuda: params.device = torch.device('cuda')
    else: params.device = torch.device('cpu')

    print(params.device)

    # Set the random seed for reproducible experiments
    torch.manual_seed(42)
    if params.cuda: torch.cuda.manual_seed(42)

    # Set the logger
    util.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dl = data_loader.fetch_dataloader(args.data_dir, 'train', params, shuffle=True)
    val_dl = data_loader.fetch_dataloader(args.data_dir, 'valid', params, shuffle=False)

    logging.info("- done.")

    # Define the model and optimizer
    g = began.BeganGenerator(params).to(device=params.device)
    d = began.BeganDiscriminator(params).to(device=params.device)
    g_optimizer = optim.Adam(g.parameters(), lr=params.g_learning_rate,
                                betas=(params.beta1,params.beta2))
    d_optimizer = optim.Adam(d.parameters(), lr=params.d_learning_rate,
                                betas=(params.beta1,params.beta2))

    # Apply weight intialization
    g.apply(began.weights_init)
    d.apply(began.weights_init)

    # fetch loss function and metrics
    metrics = began.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(g, d, train_dl, val_dl, g_optimizer, d_optimizer,
                       metrics, params, args.model_dir, args.restore_file)
