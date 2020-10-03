import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualEncoder(nn.Module):
    '''
    Reference: https://github.com/pytorch/examples/blob/master/vae/main.py
    Paper: "Hierarchical Generative Modeling for Controllable Speech Synthesis"
    '''
    def __init__(self, hparams):
        super(ResidualEncoder, self).__init__()
        self.hparams = hparams
        self.conv_in_channels = hparams.n_mel_channels * hparams.n_frames_per_step
        self.conv_out_channels = hparams.res_en_conv_kernels
        self.lstm_hidden_size = hparams.res_en_lstm_dim
        self.out_dim = hparams.res_en_out_dim
        self.conv_kernel_size = hparams.res_en_conv_kernel_size
        self.padding = self.conv_kernel_size // 2

        self.conv1d_1 = torch.nn.Conv1d(in_channels=self.conv_in_channels, out_channels=self.conv_out_channels, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.conv1d_2 = torch.nn.Conv1d(in_channels=self.conv_out_channels, out_channels=self.conv_out_channels, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.bi_lstm = torch.nn.LSTM(input_size=self.conv_out_channels,hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_proj_mean = torch.nn.Linear(in_features=2*self.lstm_hidden_size, out_features=self.out_dim, bias=False)
        self.linear_proj_logvar = torch.nn.Linear(in_features=2*self.lstm_hidden_size, out_features=self.out_dim, bias=False)

    def forward(self, inputs):
        """ Residual Encoder
        PARAMS
        ------
        inputs: torch.Tensor. size == [batch_size, freq_len, time_len]. Mel spectrograms of mini-batch samples.

        RETURNS
        -------
        z: torch.Tensor. size == [batch_size, self.out_dim]. Gaussian-sampled latent vectors of a variational autoencoder.
        """
        h_conv = F.relu(self.conv1d_1(inputs))
        out_conv = F.relu(self.conv1d_2(h_conv)) # out_conv.shape == [batch_size, 512, mel_step]
        out_conv = out_conv.permute(0, 2, 1) # lstm gets inputs of shape [batch_size, seq_len, 2*256]
        out_lstm, _ = self.bi_lstm(out_conv) # both h_0 and c_0 default to zero. out_lstm.size == [out_lstm, seq_len, 2*256]
        avg_pooled = torch.mean(out_lstm, dim=1) # avg_pooled.shape == [batch, 2*256]

        mu = self.linear_proj_mean(avg_pooled)
        log_epsilon = self.linear_proj_logvar(avg_pooled)
        var = torch.exp(log_epsilon) + self.hparams.std_lower_bound**2
        logvar = torch.log(var)

        residual_encoding = self.reparameterize(mu, logvar)

        return residual_encoding, mu, logvar

    def inference(self, batch_size, dtype=torch.float32):
        """ Residual Encoder
        PARAMS
        ------
        batch_size: int.

        RETURNS
        -------
        z: torch.Tensor. size == [batch_size, self.out_dim].
        """
        z = torch.zeros([batch_size, self.out_dim], dtype=dtype).cuda()

        return z

    def reparameterize(self, mu, logvar):
        ''' Reparameterization trick in VAE
        Reference: https://github.com/pytorch/examples/blob/master/vae/main.py
        PARAMS
        ------
        mu: mean vectors
        logvar: log variance vectors

        RETURNS
        -------
        z: latent vectors sampled from simple Gaussian distributions
        '''
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std

        return z


class LocalRefEncoder(nn.Module):
    '''
    inputs --- [N, n_mels, Ty]  mels
    outputs --- ([N, seq_len, ref_enc_gru_size])
    '''
    def __init__(self, hparams):
        super(LocalRefEncoder, self).__init__()
        self.hparams = hparams
        K = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters

        #convs = [nn.Conv2d(in_channels=filters[i],
        convs = [CoordConv2d(in_channels=filters[i],
                                out_channels=filters[i + 1],
                                kernel_size=hparams.ref_enc_filter_size,
                                stride=hparams.ref_enc_strides,
                                padding=hparams.ref_enc_pad) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i])
             for i in range(K)])

        #out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, K)
        #out_channels = hparams.n_mel_channels
        out_channels = self.calculate_channels(hparams.n_mel_channels,
            hparams.ref_enc_filter_size[1], hparams.ref_enc_strides[1],
            hparams.ref_enc_pad[1], K)

        self.gru = nn.GRU(input_size=hparams.ref_enc_filters[-1] * out_channels,
                          hidden_size=hparams.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hparams.n_mel_channels
        self.ref_enc_gru_size = hparams.ref_enc_gru_size
        self.prosody_dim = hparams.prosody_dim

        # 1dConv linear project at each step.
        self.linear_projection = ConvNorm(
            in_channels=self.ref_enc_gru_size,
            out_channels=self.prosody_dim,
            bias=True,
            w_init_gain='relu')
        self.bn_lp = nn.BatchNorm1d(self.prosody_dim)

    def forward(self, inputs, input_lengths=None):
        inputs = inputs.transpose(1, 2) #  [N, n_mels, Ty] -> [N, Ty, n_mels]
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels) # [N, Ty, n_mels] -> [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, 128, Ty, n_mels] -> [N, Ty, 128, n_mels // 2 ** mel_wise_stride]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty, 128*(n_mels // 2 ** mel_wise_stride)]

        if input_lengths is not None:
            input_lengths = (input_lengths.cpu().numpy() / 2 ** len(self.convs))
            input_lengths = input_lengths.round().astype(int)
            out = nn.utils.rnn.pack_padded_sequence(
                        out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        outputs, _ = self.gru(out)
        outputs = outputs.transpose(1, 2) #  [N, Ty, ref_enc_gru_size] -> [N, ref_enc_gru_size, Ty]
        outputs = F.relu(self.bn_lp(self.linear_projection(outputs))) # [N, ref_enc_gru_size, Ty] -> [N, prosody_dim, Ty]
        outputs = outputs.transpose(1, 2) # [N, prosody_dim, Ty] -> [N, Ty, prosody_dim]
        last_output = outputs[-1,:,:]
        #_, out = self.gru(out)
        return outputs, _
        #return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class LocalRefEncoder2CNN2biLSTM(nn.Module):
    '''
    A lower part of ResidualEncoder
    '''
    def __init__(self, hparams):
        super(LocalRefEncoder2CNN2biLSTM, self).__init__()
        self.hparams = hparams
        self.conv_in_channels = 1
        self.conv_out_channels = hparams.res_en_conv_kernels
        self.out_dim = hparams.res_en_out_dim
        self.conv_kernel_size = hparams.res_en_conv_kernel_size
        self.padding = (self.conv_kernel_size[0] // 2, self.conv_kernel_size[1] // 2)
        self.lstm_input_dim = self.conv_out_channels * hparams.n_mel_channels
        self.lstm_hidden_size = hparams.res_en_lstm_dim

        self.conv2d_1 = CoordConv2d(in_channels=self.conv_in_channels,
            out_channels=self.conv_out_channels,
            kernel_size=self.conv_kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm2d(self.conv_out_channels)
        self.conv2d_2 = CoordConv2d(in_channels=self.conv_out_channels,
            out_channels=self.conv_out_channels,
            kernel_size=self.conv_kernel_size, padding=self.padding)
        self.bn2 = nn.BatchNorm2d(self.conv_out_channels)
        self.bi_lstm = torch.nn.LSTM(input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True,
            bidirectional=True)

        # 1dConv linear project at each step.
        self.linear_projection = ConvNorm(
            in_channels=2*self.lstm_hidden_size,
            out_channels=hparams.prosody_dim,
            bias=True,
            w_init_gain='relu')
        self.bn_lp = nn.BatchNorm1d(self.prosody_dim)

    def forward(self, inputs):
        """ Residual Encoder
        PARAMS
        ------
        inputs: torch.Tensor. size == [batch_size, freq_len, time_len]. Mel spectrograms of mini-batch samples.

        RETURNS
        -------
        out_lstm: torch.Tensor. size == [batch_size, seq_len, prosody_dim]. Prosody encoding at each Mel frame
        """
        inputs = inputs.unsqueeze(1) # Conv input should be [N, C, H, W]. [batch_size, 1, freq_len, time_len]
        h_conv = F.relu(self.bn1(self.conv2d_1(inputs)))
        out_conv = F.relu(self.bn2(self.conv2d_2(h_conv))) # out_conv.shape == [batch_size, 512, mel_step]
        out_conv = out_conv.view(out_conv.size(0), -1, out_conv.size(-1))
        out_conv = out_conv.transpose(1, 2)
        out_lstm, _ = self.bi_lstm(out_conv) # both h_0 and c_0 default to zero. out_lstm.size == [batch_size, seq_len, 2*256]
        out_lstm = out_lstm.transpose(1, 2) # [batch_size, 2*lstm_hidden_size, seq_len]
        outputs = F.relu(self.bn_lp(self.linear_projection(out_lstm))) # [batch_size, prosody_dim, seq_len]
        outputs = outputs.transpose(1, 2) # [batch_size, seq_len, prosody_dim]

        return outputs


class GlocalRefEncoderRNNCell(nn.Module):
    def __init__(self, hparams):
        super(GlocalRefEncoderRNNCell, self).__init__()
        self.hparams = hparams
        K = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters

        #convs = [nn.Conv2d(in_channels=filters[i],
        convs = [CoordConv2d(in_channels=filters[i],
                                out_channels=filters[i + 1],
                                kernel_size=hparams.ref_enc_filter_size,
                                stride=hparams.ref_enc_strides,
                                padding=hparams.ref_enc_pad) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hparams.n_mel_channels,
            hparams.ref_enc_filter_size[1], hparams.ref_enc_strides[1],
            hparams.ref_enc_pad[1], K)

        # Construct GRUCell
        self.temp_gru_cell = nn.GRUCell(
            input_size=hparams.ref_enc_filters[-1] * out_channels,
            hidden_size=hparams.ref_enc_gru_size)
        self.global_gru_cell = nn.GRUCell(
            input_size=hparams.ref_enc_gru_size,
            hidden_size=hparams.ref_enc_gru_size)

        # linear projection at each step.
        self.linear1 = LinearNorm(
            in_dim=hparams.ref_enc_gru_size,
            out_dim=hparams.prosody_dim,
            bias=True,
            w_init_gain='relu')
        self.bn_l1 = nn.BatchNorm1d(hparams.prosody_dim)
        self.linear2 = LinearNorm(
            in_dim=hparams.ref_enc_gru_size,
            out_dim=hparams.prosody_dim,
            bias=True,
            w_init_gain='relu')
        self.bn_l2 = nn.BatchNorm1d(hparams.prosody_dim)

    def forward(self, inputs):
        '''
        Params
        -----

        Returns
        -----

        '''
        inputs = inputs.transpose(1, 2) #  [N, n_mels, Ty] -> [N, Ty, n_mels]
        out = inputs.view(inputs.size(0), 1, -1, self.hparams.n_mel_channels) # [N, Ty, n_mels] -> [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, 128, Ty, n_mels] -> [N, Ty, 128, n_mels // 2 ** mel_wise_stride]
        N, T = out.size(0), out.size(1)
        conv_out = out.contiguous().view(N, T, -1)  # [N, Ty, 128*(n_mels // 2 ** mel_wise_stride)]

        # Initialize GRPCells
        self.initialize_states()

        seq_len = inputs.size(1)
        temp_outputs, global_outputs = [], []

        while len(temp_outputs) < seq_len:
            t = len(temp_outputs)
            gru_input = conv_out[:,t,:]
            self.temp_gru_hidden = self.temp_gru_cell(gru_input, self.temp_gru_hidden)
            temp_gru_hidden_detached = self.temp_gru_hidden.detach()
            self.global_gru_hidden = self.global_gru_cell(temp_gru_hidden_detached, self.global_gru_hidden)
            temp_output = F.relu(self.bn_l1(self.linear1(self.temp_gru_hidden)))
            global_output = F.relu(self.bn_l2(self.linear2(self.global_gru_hidden)))
            temp_outputs += [temp_output]
            global_outputs += [global_output]

        self.temp_outputs, self.global_outputs \
            = self.parse_outputs(temp_outputs, global_outputs)

        return self.temp_outputs, self.global_outputs

    def get_gru_cell(self):
        return self.temp_gru_cell, self.global_gru_cell

    def initialize_states(self):
        self.temp_gru_hidden = None
        self.global_gru_hidden = None

    def parse_outputs(self, temp_outputs, global_outputs):
        temp_outputs = torch.stack(temp_outputs)
        temp_outputs = temp_outputs.transpose(0,1) # (T_out, B, gru_dim) -> (B, T_out, prosody_dim)
        global_outputs = torch.stack(global_outputs)
        global_outputs = global_outputs.transpose(0,1) # (T_out, B, gru_dim) -> (B, T_out, prosody_dim)

        return temp_outputs, global_outputs

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class GlocalRefEncoder(nn.Module):
    '''
    inputs --- [N, n_mels, Ty]  mels
    outputs --- ([N, seq_len, ref_enc_gru_size])
    '''
    def __init__(self, hparams):
        super(GlocalRefEncoder, self).__init__()
        self.hparams = hparams
        K = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters

        #convs = [nn.Conv2d(in_channels=filters[i],
        convs = [CoordConv2d(in_channels=filters[i],
                                out_channels=filters[i + 1],
                                kernel_size=hparams.ref_enc_filter_size,
                                stride=hparams.ref_enc_strides,
                                padding=hparams.ref_enc_pad) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i])
             for i in range(K)])

        #out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, K)
        #out_channels = hparams.n_mel_channels
        out_channels = self.calculate_channels(hparams.n_mel_channels,
            hparams.ref_enc_filter_size[1], hparams.ref_enc_strides[1],
            hparams.ref_enc_pad[1], K)

        self.temp_gru = nn.GRU(input_size=hparams.ref_enc_filters[-1] * out_channels,
                          hidden_size=hparams.ref_enc_gru_size,
                          batch_first=True)
        self.h_temp_gru = None

        self.global_gru = nn.GRU(input_size=hparams.ref_enc_gru_size,
                          hidden_size=hparams.ref_enc_gru_size,
                          batch_first=True)
        self.h_global_gru = None

        # 1dConv linear project at each step.
        self.linear_conv = ConvNorm(
            in_channels=hparams.ref_enc_gru_size,
            out_channels=hparams.prosody_dim,
            bias=True,
            w_init_gain='relu')
        self.bn_l = nn.BatchNorm1d(hparams.prosody_dim)

    def initialize_states(self):
        self.h_temp_gru = None
        self.h_global_gru = None

    def forward(self, inputs, input_lengths=None):
        inputs = inputs.transpose(1, 2) #  [N, n_mels, Ty] -> [N, Ty, n_mels]
        out = inputs.view(inputs.size(0), 1, -1, self.hparams.n_mel_channels) # [N, Ty, n_mels] -> [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, 128, Ty, n_mels] -> [N, Ty, 128, n_mels // 2 ** mel_wise_stride]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty, 128*(n_mels // 2 ** mel_wise_stride)]

        if input_lengths is not None:
            input_lengths = (input_lengths.cpu().numpy() / 2 ** len(self.convs))
            input_lengths = input_lengths.round().astype(int)
            out = nn.utils.rnn.pack_padded_sequence(
                        out, input_lengths, batch_first=True, enforce_sorted=False)

        conv_out = out
        self.temp_gru.flatten_parameters()
        self.global_gru.flatten_parameters()

        out_temp_gru, self.h_temp_gru = self.temp_gru(conv_out, self.h_temp_gru)
        out_temp_gru_detached = out_temp_gru.detach()
        out_global_gru, self.h_global_gru = self.global_gru(out_temp_gru_detached, self.h_global_gru)

        out_temp_gru = out_temp_gru.transpose(1, 2) #  [N, Ty, ref_enc_gru_size] -> [N, ref_enc_gru_size, Ty]
        out_global_gru = out_global_gru.transpose(1, 2) #  [N, Ty, ref_enc_gru_size] -> [N, ref_enc_gru_size, Ty]

        if self.hparams.global_prosody_is_hidden:
            out_temp = F.relu(self.bn_l(self.linear_conv(out_temp_gru))) # [N, ref_enc_gru_size, Ty] -> [N, prosody_dim, Ty]
            out_temp = out_temp.transpose(1,2) # [N, prosody_dim, Ty] -> [N, Ty, prosody_dim]
            out_global_gru = out_global_gru.transpose(1, 2) # [N, ref_enc_gru_size, Ty] -> [N, Ty, ref_enc_gru_size]
            return out_temp, out_global_gru

        else:
            out_cat = torch.cat([out_temp_gru, out_global_gru], dim=0) # [2*N, ref_enc_gru_size, Ty]
            out_cat = F.relu(self.bn_l(self.linear_conv(out_cat))) # [2*N, ref_enc_gru_size, Ty] -> [2*N, prosody_dim, Ty]

            out_temp = out_cat[:N,:,:].transpose(1,2) # [2*N, prosody_dim, Ty] -> [N, Ty, prosody_dim]
            out_global = out_cat[N:,:,:].transpose(1,2) # [2*N, prosody_dim, Ty] -> [N, Ty, prosody_dim]

            return out_temp, out_global

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
