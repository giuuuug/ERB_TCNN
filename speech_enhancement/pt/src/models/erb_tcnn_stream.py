import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch


###########################################################
# Streaming ERB-TCNN blocks and layers
###########################################################

class DepthwiseSeparableConvStream(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, layer_activation="relu"):
        super(DepthwiseSeparableConvStream, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.receptive_field = (kernel_size - 1) * dilation

        if layer_activation == "prelu":
            act = nn.PReLU()
        elif layer_activation == "relu":
            act = nn.ReLU()

        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=0, dilation=dilation, groups=in_channels, bias=False
        )
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = act
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, state):
        # x: [Batch, Canali, Tempo=1]
        # state: [Batch, Canali, Memoria_Passata(RF)]
        
        if self.receptive_field > 0:
            x_concat = torch.cat([state, x], dim=2)  # [B, C, RF + T]
        else:
            x_concat = x

        out = self.depthwise_conv(x_concat)
        out = self.bn(out)
        out = self.act(out)
        out = self.pointwise_conv(out)
        
        if self.receptive_field > 0:
            new_state = x_concat[:, :, -self.receptive_field:]
        else:
            new_state = state

        return out, new_state


class ResBlockStream(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, layer_activation="relu"):
        super(ResBlockStream, self).__init__()
        if layer_activation == "prelu":
            self.act = nn.PReLU(num_parameters=1)
        elif layer_activation == "relu":
            self.act = nn.ReLU()

        self.conv1x1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.dws_conv = DepthwiseSeparableConvStream(
            in_channels=out_channels, out_channels=in_channels, 
            kernel_size=kernel_size, stride=1, dilation=dilation, layer_activation=layer_activation
        )

    def forward(self, input, state):
        x = self.conv1x1(input)
        x = self.bn(x)
        x = self.act(x)
        x, new_state = self.dws_conv(x, state)
        return x + input, new_state


class TCNN_BlockStream(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=2, num_layers=5, layer_activation="relu"):
        super(TCNN_BlockStream, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation_size = init_dilation ** i
            self.layers.append(
                ResBlockStream(in_channels, out_channels, kernel_size, dilation=dilation_size, layer_activation=layer_activation)
            )

    def forward(self, x, states):
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state_out = layer(x, states[i])
            new_states.append(state_out)
        return x, new_states


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
        x_low = x[:, :self.erb_subband_1, :]
        x_high = x[:, self.erb_subband_1:, :]
        x_erb = self.erb_fc(x_high)
        return torch.cat([x_low, x_erb], dim=1)

    def bs(self, x):
        x_low = x[:, :self.erb_subband_1, :]
        x_erb = x[:, self.erb_subband_1:, :]
        x_high = self.ierb_fc(x_erb)
        return torch.cat([x_low, x_high], dim=1)


class ERBTCNN_Stream(nn.Module):
    def __init__(self, tcn_latent_dim=512, in_channels=257, n_blocks=2, kernel_size=3, num_layers=5,
                 mask_activation="tanh", layer_activation="relu", init_dilation=2,
                 erb_subband_1=65, erb_subband_2=64, nfft=512, fs=16000,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.mask_activation = mask_activation
        self.layer_activation = layer_activation
        self.init_dilation = init_dilation
        
        self.erb = ERB(erb_subband_1, erb_subband_2, nfft, high_lim=fs//2, fs=fs)
        self.tcn_channels = erb_subband_1 + erb_subband_2

        self.tcn_blocks = nn.ModuleList()
        for k in range(n_blocks):
            self.tcn_blocks.append(TCNN_BlockStream(
                           in_channels=self.tcn_channels,
                           out_channels=self.tcn_channels,
                           kernel_size=self.kernel_size,
                           init_dilation=self.init_dilation,
                           num_layers=self.num_layers,
                           layer_activation=self.layer_activation))
            
        if self.mask_activation == "tanh":
            self.activation = nn.Tanh()
        elif self.mask_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("mask_activation must be one of ['tanh', 'sigmoid']")
        
    def forward(self, input, states):
        x = self.erb.bm(input)
        
        new_states = []
        for i, block in enumerate(self.tcn_blocks):
            x, block_states = block(x, states[i])
            new_states.append(block_states)
        
        x = self.erb.bs(x)
        out = self.activation(x)
        return out, new_states

    def get_initial_states(self, batch_size=1, device='cpu'):
        states = []
        for k in range(self.n_blocks):
            block_states = []
            for i in range(self.num_layers):
                dilation_size = self.init_dilation ** i
                rf = (self.kernel_size - 1) * dilation_size
                block_states.append(torch.zeros(batch_size, self.tcn_channels, rf, device=device))
            states.append(block_states)
        return states
