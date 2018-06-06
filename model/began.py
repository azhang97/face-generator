'''
began.py

Based on https://github.com/pytorch/examples/blob/master/dcgan/main.py
BEGAN Model https://arxiv.org/pdf/1703.10717.pdf
'''


import torch
import torch.nn as nn
import model.model_util as model_util

# From https://github.com/pytorch/examples/blob/master/dcgan/main.py
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BeganGenerator(nn.Module):
    def __init__(self, params):
        super(BeganGenerator, self).__init__()
        self.params = params
        self.z_fixed = torch.FloatTensor(params.num_sample_imgs, params.h).uniform_(-1,1).to(params.device)
        self.main = nn.Sequential(
            # Dim: batch_size x h
            nn.Linear(self.params.h, self.params.n * 8 * 8),
            # Dim: batch_size x (n * 8 * 8)
            model_util.View(-1, self.params.n, 8, 8),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, 3, kernel_size=3, stride=1, padding=1),
            # Dim: batch_size x 3 (RGB Channels) x 128 x 128
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.params.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.opt['ngpu']))
        else:
            output = self.main(input)
        return output

    def loss_fn(self, g_img, g_img_passed):
        return torch.mean(torch.abs(g_img_passed - g_img))


class BeganDiscriminator(nn.Module):
    def __init__(self, params):
        super(BeganDiscriminator, self).__init__()
        self.params = params
        self.began_k = 0
        self.encoder = nn.Sequential(
            ########## ENCODER ##########
            # Dim: batch_size x 3 (RGB Channels) x 128 x 128
            nn.Conv2d(3, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 8 x 8
            model_util.View(-1, self.params.n * 8 * 8),
            # Dim: batch_size x (n * 8 * 8)
            nn.Linear(self.params.n * 8 * 8, self.params.h),
            # Dim: batch_size x h
        )
        self.decoder = nn.Sequential(
            ########## DENCODER ##########
            # Dim: batch_size x h
            nn.Linear(self.params.h, self.params.n * 8 * 8),
            # Dim: batch_size x (n * 8 * 8)
            model_util.View(-1, self.params.n, 8, 8),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 8 x 8
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 16 x 16
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 32 x 32
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 64 x 64
            nn.Upsample(scale_factor=2),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, self.params.n, kernel_size=3, stride=1, padding=1),
            nn.ELU(self.params.elu_alpha, inplace=True),
            # Dim: batch_size x n x 128 x 128
            nn.Conv2d(self.params.n, 3, kernel_size=3, stride=1, padding=1),
            # Dim: batch_size x 3 (RGB Channels) x 128 x 128
            nn.Tanh()
        )


    def forward(self, input):
        if input.is_cuda and self.params.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.params.ngpu))
        else:
            h = self.encoder(input)
            output = self.decoder(h)
        return output

    def loss_fn(self, r_img, g_img, r_img_passed, g_img_passed):
        d_r_loss = torch.mean(torch.abs(r_img_passed - r_img))
        d_g_loss = torch.mean(torch.abs(g_img_passed - g_img))
        d_loss = d_r_loss - self.began_k * d_g_loss

        # Update began_k value
        balance = (self.params.began_gamma * d_r_loss - d_g_loss).data
        self.began_k = min(max(self.began_k + self.params.began_lambda_k * balance, 0), 1)
        return d_loss


# Metrics
def began_convergence(r_img, g_img, r_img_passed, g_img_passed, began_gamma):
    d_r_loss = torch.mean(torch.abs(r_img_passed - r_img))
    d_g_loss = torch.mean(torch.abs(g_img_passed - g_img))

    return d_r_loss + torch.abs(began_gamma * d_r_loss - d_g_loss)

metrics = {
    #'began_convergence':began_convergence
}
