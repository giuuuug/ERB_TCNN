import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch


###########################################################
# TCNN blocks and layers, taken straight from the TCNN repo
# https://github.com/LXP-Never/TCNN
###########################################################

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation,
                  causal=False, layer_activation="relu"):
        super(DepthwiseSeparableConv, self).__init__()
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = dilation

        if layer_activation == "prelu":
            act = nn.PReLU()
        elif layer_activation == "relu":
            act = nn.ReLU()

            
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=False)

        pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        if causal:
            self.net = nn.Sequential(depthwise_conv,
                                     Chomp1d(padding),
                                     nn.BatchNorm1d(in_channels),
                                     act,
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv,
                                     nn.BatchNorm1d(in_channels),
                                     act,
                                     pointwise_conv)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, layer_activation="relu"):
        super(ResBlock, self).__init__()
        if layer_activation == "prelu":
            act = nn.PReLU(num_parameters=1)
        elif layer_activation == "relu":
            act = nn.ReLU()

        self.TCM_net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            act,
            DepthwiseSeparableConv(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=1, dilation=dilation, causal=False, layer_activation=layer_activation)
        )

    def forward(self, input):
        x = self.TCM_net(input)
        return x + input


class TCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=3, num_layers=6,
                 layer_activation="relu"):
        super(TCNN_Block, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation ** i

            layers += [ResBlock(in_channels, out_channels,
                                kernel_size, dilation=dilation_size,
                                layer_activation=layer_activation)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        
        self.erb_fc = nn.Conv1d(nfreqs - erb_subband_1, erb_subband_2, kernel_size=1, bias=False)
        self.erb_fc.weight.data = torch.from_numpy(erb_filters).float().unsqueeze(2)
        self.erb_fc.weight.requires_grad = False
        
        self.ierb_fc = nn.Conv1d(erb_subband_2, nfreqs - erb_subband_1, kernel_size=1, bias=False)
        self.ierb_fc.weight.data = torch.from_numpy(erb_filters.T).float().unsqueeze(2)
        self.ierb_fc.weight.requires_grad = False

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2 + 2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        for i in range(erb_subband_2):
            for j in range(bins[i], bins[i + 1]):
                erb_filters[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
            for j in range(bins[i + 1], bins[i + 2]):
                erb_filters[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])
        
        return erb_filters[:, erb_subband_1:]

    def bm(self, x):
        """Forward through ERB compression. x shape: (B, Freqs, T)"""
        x_low = x[:, :self.erb_subband_1, :]
        x_high = x[:, self.erb_subband_1:, :]
        x_erb = self.erb_fc(x_high)
        return torch.cat([x_low, x_erb], dim=1)

    def bs(self, x):
        """Inverse through ERB expansion. x shape: (B, Freqs, T)"""
        x_low = x[:, :self.erb_subband_1, :]
        x_erb = x[:, self.erb_subband_1:, :]
        x_high = self.ierb_fc(x_erb)
        return torch.cat([x_low, x_high], dim=1)


class ERBTCNN(nn.Module):
    '''
    ERB-TCNN model : TCNN-type denoising network without the convolutional encoder and decoders
    , using ERB preprocessing instead.
    Takes magnitude spectrogram columns as input, and outputs columns of a mask to be applied 
    to the complex spectrogram.
    Input shape is (batch, in_channels, sequence_length)
    Output shape is (batch, in_channels, sequence_length)
    '''
    def __init__(self,
                 in_channels=257,
                 tcn_latent_dim=512,
                 n_blocks=3,
                 kernel_size=3,
                 num_layers=6,
                 mask_activation="tanh",
                 layer_activation="relu",
                 init_dilation=2,
                 erb_subband_1=2,
                 erb_subband_2=64,
                 nfft=512,
                 fs=16000,
                 **kwargs
                 ):
        '''
        Parameters
        ----------
        in_channels, int : Number of channels in input and output tensors. Should be n_fft // 2 + 1
        tcn_latent_dim, int : Latent dimension of the temporal convolutional blocks
        n_blocks, int : Number of temporal convolutional blocks
        n_layers, int : Number of layers per temporal convolutional block
        mask_activation, str : One of "tanh", "sigmoid". Final activation applied to model output.
            "tanh" means output coeffs are in [-1, 1], and "sigmoid" means they are in [0, 1]
        erb_subband_1, int : Number of linearly spaced channels to leave untouched by ERB compression
        erb_subband_2, int : Number of output filters calculated dynamically in ERB compression
        nfft, int : STFT size
        fs, int : Sample Rate
        NOTE : Additional kwargs are discarded, this is for convenience to allow user to pass
        invalid model-specific kwargs without raising an exceptions
        '''
        super().__init__()
        self.in_channels = in_channels
        self.tcn_latent_dim = tcn_latent_dim
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.mask_activation = mask_activation
        self.layer_activation = layer_activation
        self.init_dilation = init_dilation
        
        self.erb = ERB(erb_subband_1, erb_subband_2, nfft, high_lim=fs//2, fs=fs)
        self.tcn_channels = erb_subband_1 + erb_subband_2

        self.tcn_block_dict = OrderedDict()
        for k in range(n_blocks):
            self.tcn_block_dict[f"tcn_block_{k}"] = TCNN_Block(
                           in_channels=self.tcn_channels,
                           out_channels=self.tcn_channels,
                           kernel_size=self.kernel_size,
                           init_dilation=self.init_dilation,
                           num_layers=self.num_layers,
                           layer_activation=self.layer_activation)
            
        self.tcn = nn.Sequential(self.tcn_block_dict)
        if self.mask_activation == "tanh":
            self.activation = nn.Tanh()
        elif self.mask_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("mask_activation must be one of ['tanh', 'sigmoid']")
        
    def forward(self, input):
        # No encoder or decoder
        # Forward through ERB compression
        x = self.erb.bm(input)
        
        # Process in TCNN domain
        x = self.tcn(x)
        
        # Expand out of ERB domain
        x = self.erb.bs(x)
        
        out = self.activation(x)
        return out