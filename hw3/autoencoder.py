import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======

        # (Conv -> Bnorm -> ReLU)*3 + (linear -> Bnorm -> ReLU) :
        filters = [in_channels] + [64, 128, 256] + [out_channels]

        for i in range(1, len(filters)):
            in_chann = filters[i - 1]
            out_chann = filters[i]
            modules.append(nn.Conv2d(in_channels=in_chann, out_channels=out_chann, kernel_size=4, padding=1, stride=2))
            modules.append(nn.BatchNorm2d(out_chann))
            modules.append(nn.ReLU())

        # modules.append(nn.Linear(2048, out_channels))
        # modules.append(nn.ReLU())
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======

        filters = [in_channels] + [64, 128, 256] + [out_channels]
        
        for i in range(1, len(filters)):
            in_chann = filters[i-1]
            out_chann = filters[i]
            modules.append(nn.ConvTranspose2d(in_channels=in_chann, out_channels=out_chann, kernel_size=4, padding=1, stride=2))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm2d(out_chann))

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)
        #print(n_features,self.features_shape)

        # TODO: Add parameters needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.m_alpha = nn.Linear(n_features, z_dim)
        self.log_sigma_alpha = nn.Linear(n_features, z_dim)
        self.m_beta = nn.Linear(z_dim, n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO: Sample a latent vector z given an input x.
        # 1. Use the features extracted from the input to obtain mu and
        # log_sigma2 (mean and log variance) of the posterior p(z|x).
        # 2. Apply the reparametrization trick.
        # ====== YOUR CODE: ======

        h = self.features_encoder.forward(x)
        h = h.view(h.size(0), -1)
        u = torch.distributions.normal.Normal(0, 1).sample()

        mu = self.m_alpha(h)
        log_sigma2 = self.log_sigma_alpha(h)
        z = mu + u*log_sigma2.exp()    # todo: devide by 2?
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO: Convert a latent vector back into a reconstructed input.
        # 1. Convert latent to features.
        # 2. Apply features decoder.
        # ====== YOUR CODE: ======
        h_rec = self.m_beta(z)
        h_rec = h_rec.view(-1, self.features_shape[0], self.features_shape[1], self.features_shape[1])
        x_rec = self.features_decoder(h_rec)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO: Sample from the model.
            # Generate n latent space samples and return their reconstructions.
            # Remember that for the model, this is like inference.
            # ====== YOUR CODE: ======
            z = torch.randn(n, self.z_dim, device=device)
            samples = self.decode(z)
            # ========================
        return samples.to('cpu')

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO: Implement the VAE pointwise loss calculation.
    # Remember:
    # 1. The covariance matrix of the posterior is diagonal.
    # 2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======

    N, C, H, W = x.shape
    dz = z_mu.shape[1]

    data_loss = ((x - xr).norm() ** 2 / x_sigma2) / (N*C*H*W)
    kldiv_loss = (z_log_sigma2.exp().sum() + z_mu.norm()**2 - dz*N - z_log_sigma2.sum()) / N
    loss = data_loss + kldiv_loss

    # ========================

    return loss, data_loss, kldiv_loss
