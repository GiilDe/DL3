from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======

        C_in, H_in, W_in  = in_size

        filters = [C_in] + [64, 128, 256]
        modules = []

        for i in range(1, len(filters)):
            in_chann = filters[i - 1]
            out_chann = filters[i]
            modules.append(nn.Conv2d(in_channels=in_chann, out_channels=out_chann, kernel_size=5, padding=1, stride=1))
            modules.append(nn.BatchNorm2d(out_chann))
            modules.append(nn.ReLU())
        self.conv = nn.Sequential(*modules)

        modules = []

        #H_out = (H_in +2*1 -1 // 1 ) + 1
        #W_out = (W_in +2*1 -1 // 1 ) + 1
        #C_out = 256


        modules.append(nn.Linear(861184, 1))
        #modules.append(nn.Linear(16384, 1))
        modules.append(nn.ReLU())
        self.linear = nn.Sequential(*modules)

        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======

        y = self.conv(x)
        y = y.view(y.size(0), -1)
        y = self.linear(y)

        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======

        modules = []
        filters = [z_dim] + [250, 500, 750, 1000] + [out_channels]

        for i in range(1, len(filters)):
            in_chann = filters[i - 1]
            out_chann = filters[i]
            modules.append(
                nn.ConvTranspose2d(in_channels=in_chann, out_channels=out_chann, kernel_size=featuremap_size,
                                   padding=1 if in_chann != self.z_dim else 0, stride=2))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm2d(out_chann))

        self.cnn = nn.Sequential(*modules)

        # self.linear = nn.Linear(z_dim, 512*4)
        #self.dec = DecoderCNN(512, out_channels)
        #self.last_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======

        with torch.set_grad_enabled(with_grad):
            gauss = torch.distributions.normal.Normal(0,1)
            latent_space_samples = gauss.sample(sample_shape = (n, self.z_dim)).to(device)
            samples = self.forward(latent_space_samples).to(device)
        # ========================
        return samples


    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        '''
        N = z.shape[0]
        z = self.linear(z)
        z = z.reshape(N, 512, 2, 2)
        z = self.dec(z)
        x = self.last_conv(z)
        '''
        z = torch.unsqueeze(z, dim=2)
        z = torch.unsqueeze(z, dim=3)
        x = self.cnn(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    device = y_data.device
    noise = torch.distributions.uniform.Uniform(-label_noise/2, label_noise/2)
    data_noise = noise.sample(y_data.size()).to(device)
    y_data_noisy = (torch.full(y_data.size(), data_label, device=device) + data_noise).to(device)
    gen_label = 1 - data_label
    gen_noise = noise.sample(y_data.size()).to(device)
    y_gen_noisy = (torch.full(y_data.size(), gen_label, device=device) + gen_noise).to(device)
    loss_data = torch.nn.functional.binary_cross_entropy_with_logits(y_data, y_data_noisy)
    loss_generated = torch.nn.functional.binary_cross_entropy_with_logits(y_generated, y_gen_noisy)
    return loss_data + loss_generated

# loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_data = loss_fn(y_data_noisy, y_data)s
# loss_generated = loss_fn(y_gen_noisy, y_generated)

def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_generated, torch.full(y_generated.size(), data_label, device=y_generated.device))
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters

    # ====== YOUR CODE: ======

    N = x_data.shape[0]  # batch size
    generated_samples = gen_model.sample(N, with_grad=True)
    y_data = dsc_model(x_data)
    y_generated = dsc_model(generated_samples.detach())
    dsc_optimizer.zero_grad()
    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======

    y_generated = dsc_model(generated_samples)
    gen_optimizer.zero_grad()
    gen_loss = gen_loss_fn(y_generated)
    gen_loss.backward()
    gen_optimizer.step()

    # ========================

    return dsc_loss.item(), gen_loss.item()
