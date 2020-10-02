from math import sqrt
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
import time
from utils import get_spk_adv_inputs, get_emo_adv_inputs
from utils import hard_clip, soft_clip
from utils import discretize_att_w
from coordconv import CoordConv2d
from measures import SecondStopPredictor
from modules import ReferenceEncoder
from ref_encoders import ResidualEncoder, LocalRefEncoder, GlocalRefEncoder


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
            attention_weights_cat, mask, discrete_attention_weight=False):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        if ((discrete_attention_weight and isinstance(discrete_attention_weight, bool))
            or isinstance(discrete_attention_weight, list)):
            # size == (batch, max_text_step)
            attention_weights = discretize_att_w(attention_weights,
                discrete_attention_weight)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class MontonicAttention(nn.Module):
    def __init__(self, hparams, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(MontonicAttention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.mean_layer = LinearNorm(attention_dim, hparams.n_mean_units,
            bias=False, w_init_gain='sigmoid')
        self.logvar_layer = LinearNorm(attention_dim, 1, bias=False, w_init_gain='linear')

        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = 0

        self.prev_means = None
        self.prev_logvars = None
        self.logvar_min = np.log(0.1**2)
        self.logvar_max = np.log(36.770**2)


    def get_means(self):
        return self.prev_means


    def normal_pdf(self, batch_txt_length, means, stds):
        '''
        PARAMS
        -----
        batch_txt_length: int.
        means: torch.Tensor.
        - size: [batch_size]
        stds: torch.Tensor.
        - size: [batch_size]

        RETURNS
        -----
        p: torch.Tensor.
        - size: [batch_size, batch_txt_length]
        '''
        enc_steps = batch_txt_length
        batch_size = means.size(0)

        means = means.unsqueeze(1).expand(means.size(0), enc_steps)
        stds = stds.unsqueeze(1).expand(stds.size(0), enc_steps)

        x = torch.Tensor(np.arange(enc_steps).reshape((1, enc_steps))).cuda()
        x = x.expand(batch_size, enc_steps)

        p = Normal(means, stds).cdf(x+0.5) - Normal(means, stds).cdf(x-0.5)
        #p = Normal(means, stds).log_prob(x).exp()
        # p_sum is a normalizing factor to make the sum across the encoding dimension 1.
        p_sum = p.sum(dim=1, keepdim=True).expand(p.size())
        '''print("means.max()", means.max())
        print("means.max()", means.min())
        print("stds.max()", stds.max())
        print("stds.max()", stds.max())
        print("p_sum.min()", p_sum.min().item())
        print("p_sum.max()", p_sum.max().item())'''
        #p = p / p_sum

        return p

    def reset_initial_state(self):
        self.prev_means = None

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output (batch, n_mel_channels * n_frames_per_step)
        memory: encoder outputs (B, T_in, attention_dim)
        processed_memory: processed encoder outputs (B, T_in, encoder_embedding_dim)
        attention_weights_cat: previous and cummulative attention weights (B, 2, max_time)
        mask: binary mask for padded data
        """

        processed_query = self.query_layer(attention_hidden_state.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        pred_features = torch.tanh(processed_query + processed_attention_weights + processed_memory)
        # pred_features.size == (B, T_in, attention_dim)

        # Average pooling across the second dimension
        # avgpooled.size == (B, attention_dim)
        avgpooled = pred_features.mean(dim=1)

        # mean_increments.size == (B, 10)
        #mean_increments = F.sigmoid(self.mean_layer(avgpooled))
        #mean_increments = F.relu(self.mean_layer(avgpooled))
        #mean_increments = hard_clip(self.mean_layer(avgpooled))
        mean_increments = soft_clip(self.mean_layer(avgpooled))
        # mean_increment.size == (B)
        mean_increment = mean_increments.sum(dim=-1)
        if self.prev_means is None:
            self.prev_means = torch.zeros_like(mean_increment).cuda()

        means = self.prev_means + mean_increment
        self.prev_means = means

        # stds.size == (B)
        logvars = (self.logvar_max - self.logvar_min) \
            * F.sigmoid(self.logvar_layer(avgpooled)) \
             + self.logvar_min
        '''logvars = F.sigmoid(self.logvar_layer(avgpooled))'''

        stds = logvars.squeeze(-1).exp().sqrt()
        self.prev_vars = logvars

        batch_txt_length = memory.size(1)
        attention_weights = self.normal_pdf(batch_txt_length, means, stds)

        if mask is not None:
            attention_weights.data.masked_fill_(mask, self.score_mask_value)
        '''print("pred_features.size()", pred_features.size())
        print("memory.size()", memory.size())
        print("attention_weights.size()", attention_weights.size())
        print("avgpooled.size()", avgpooled.size())
        print("mean_increment.size()", mean_increment.size())
        print("means.size()", means.size())'''
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        #print("attention_weights")
        #print(attention_weights)
        #print("attention_context")
        #print(attention_context)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.hparams = hparams
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.has_style_token_lstm_1 = hparams.has_style_token_lstm_1
        self.has_style_token_lstm_2 = hparams.has_style_token_lstm_2
        self.monotonic_attention = hparams.monotonic_attention
        self.prosody_dim = self.hparams.prosody_dim
        self.pp_lstm_hidden_dim = hparams.pp_lstm_hidden_dim

        self.attention_rnn_input_dim = self.get_attention_rnn_input_dim()
        self.decoder_rnn_input_dim = self.get_decoder_rnn_input_dim()

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            self.attention_rnn_input_dim,
            hparams.attention_rnn_dim)

        if self.monotonic_attention:
            self.attention_layer = MontonicAttention(
                hparams,
                hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)
        else:
            self.attention_layer = Attention(
                hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            self.decoder_rnn_input_dim,
            hparams.decoder_rnn_dim, 1)

        if hparams.prosody_predictor:
            self.prosody_dim = hparams.prosody_dim
            if hparams.prosody_predictor == 'MLP':
                self.prosody_predictor = ProsodyPredictorMLP(hparams)
            elif hparams.prosody_predictor == 'LSTM':
                self.prosody_predictor = ProsodyPredictorLSTMCell(hparams)
            else:
                print("Incorrect input for prosody_predictor.")
                exit()

        self.linear_projection = LinearNorm(
            self.get_linear_input_dim(),
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            self.get_linear_input_dim(), 1,
            bias=True, w_init_gain='sigmoid')

        self.stop_predictor2 = SecondStopPredictor(hparams.max_decoder_steps)

    def add_ref_enc(self, reference_encoder):
        self.reference_encoder = reference_encoder

    def add_temp_prosody_decoder(self, temp_prosody_decoder):
        self.temp_prosody_decoder = temp_prosody_decoder

    def get_attention_rnn_input_dim(self):
        attention_rnn_input_dim = (self.hparams.prenet_dim
            + self.hparams.encoder_embedding_dim)

        if self.hparams.has_style_token_lstm_1:
            if self.hparams.reference_encoder:
                attention_rnn_input_dim += self.hparams.prosody_dim
            else:
                if len(self.hparams.all_speakers) > 1:
                    attention_rnn_input_dim += self.hparams.speaker_embedding_dim
                if len(self.hparams.all_emotions) > 1:
                    attention_rnn_input_dim += self.hparams.emotion_embedding_dim
                if self.hparams.residual_encoder:
                    attention_rnn_input_dim += self.hparams.res_en_out_dim

        return attention_rnn_input_dim

    def get_decoder_rnn_input_dim(self):
        decoder_rnn_input_dim = (self.hparams.attention_rnn_dim
            + self.hparams.encoder_embedding_dim)

        if self.hparams.has_style_token_lstm_2:
            if self.hparams.reference_encoder:
                decoder_rnn_input_dim += self.hparams.prosody_dim
            else:
                if len(self.hparams.all_speakers) > 1:
                    decoder_rnn_input_dim += self.hparams.speaker_embedding_dim
                if len(self.hparams.all_emotions) > 1:
                    decoder_rnn_input_dim += self.hparams.emotion_embedding_dim
                if self.hparams.residual_encoder:
                    decoder_rnn_input_dim += self.hparams.res_en_out_dim

        return decoder_rnn_input_dim

    def get_linear_input_dim(self):
        linear_input_dim = (self.hparams.decoder_rnn_dim
            + self.hparams.encoder_embedding_dim)

        if self.hparams.has_style_token_linear:
            if self.hparams.reference_encoder:
                linear_input_dim += self.hparams.prosody_dim
            else:
                if len(self.hparams.all_speakers) > 1:
                    linear_input_dim += self.hparams.speaker_embedding_dim
                if len(self.hparams.all_emotions) > 1:
                    linear_input_dim += self.hparams.emotion_embedding_dim
                if self.hparams.residual_encoder:
                    linear_input_dim += self.hparams.res_en_out_dim

        return linear_input_dim

    def get_attention_means(self):
        if self.monotonic_attention:
            att_means = self.attention_layer.get_means()
        else:
            att_means = None
        return att_means

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def get_go_prosody(self, global_prosody):
        B = global_prosody.size(0)
        prosody_dim = global_prosody.size(2)
        go_prosody = Variable(global_prosody.data.new(
            B, prosody_dim).zero_())
        return go_prosody

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        if self.hparams.reference_encoder:
            self.prosody_encoding = Variable(memory.data.new(
                B, self.prosody_dim).zero_())
        else:
            self.prosody_encoding = None
        if self.hparams.prosody_predictor == 'LSTM':
            self.prosody_hidden = Variable(memory.data.new(
                B, self.pp_lstm_hidden_dim).zero_())
            self.prosody_cell = Variable(memory.data.new(
                B, self.pp_lstm_hidden_dim).zero_())

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

        if self.monotonic_attention:
            self.attention_layer.reset_initial_state()

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments,
            attention_contexts, prosody_encodings, prosody_preds):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B, context_dim) -> (B, T_out, context_dim)
        attention_contexts = torch.stack(attention_contexts).transpose(0, 1)
        # (T_out, B, input_length) -> (B, T_out, input_length)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        if prosody_encodings[0] is not None:
            # (T_out, B, prosody_dim) -> (B, T_out, prosody_dim)
            prosody_encodings = torch.stack(prosody_encodings).transpose(0, 1)
        else:
            prosody_encodings = None
        if prosody_preds[0] is not None:
            # (T_out, B, prosody_dim) -> (B, T_out, prosody_dim)
            prosody_preds = torch.stack(prosody_preds).transpose(0, 1)
        else:
            prosody_preds = None


        return mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds

    def decode(self, decoder_input, speaker_embedding, emotion_embedding,
            residual_encoding, t_prosody_ref=None, prev_global_prosody_ref=None,
            discrete_attention_weight=False):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_inputs = [decoder_input, self.attention_context]
        if self.has_style_token_lstm_1:
            if self.hparams.reference_encoder:
                cell_inputs.append(self.prosody_encoding)
            else:
                if len(self.hparams.all_speakers) > 1:
                    cell_inputs.append(speaker_embedding)
                if len(self.hparams.all_emotions) > 1:
                    cell_inputs.append(emotion_embedding)
                if self.hparams.residual_encoder:
                    cell_inputs.append(residual_encoding)
        cell_input = torch.cat(cell_inputs, -1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask, discrete_attention_weight)

        self.attention_weights_cum += self.attention_weights

        if self.hparams.prosody_predictor:
            detached_attention_context = self.attention_context.detach()
            pp_inputs = [detached_attention_context]
            if len(self.hparams.all_speakers) > 1:
                pp_inputs.append(speaker_embedding)
            if len(self.hparams.all_emotions) > 1:
                pp_inputs.append(emotion_embedding)
            if 'prev_global_prosody' in self.hparams.pp_opt_inputs:
                pp_inputs.append(prev_global_prosody_ref)
            if 'AttRNN' in self.hparams.pp_opt_inputs:
                pp_inputs.append(self.attention_hidden.detach())
            pp_input = torch.cat(pp_inputs, -1)
            if self.hparams.prosody_predictor == 'MLP':
                prosody_pred = self.prosody_predictor(pp_input)
            elif self.hparams.prosody_predictor == 'LSTM':
                prosody_pred, self.prosody_hidden, self.prosody_cell = self.prosody_predictor(
                    pp_input, (self.prosody_hidden, self.prosody_cell))
        else:
            prosody_pred = None

        if not self.hparams.reference_encoder:
            pass
        elif (self.hparams.prosody_predictor != '') and (not self.training):
            self.prosody_encoding = prosody_pred
        else:
            self.prosody_encoding = t_prosody_ref

        decoder_inputs = [self.attention_hidden, self.attention_context]
        if self.has_style_token_lstm_2:
            if self.hparams.reference_encoder:
                decoder_inputs.append(self.prosody_encoding)
            else:
                if len(self.hparams.all_speakers) > 1:
                    decoder_inputs.append(speaker_embedding)
                if len(self.hparams.all_emotions) > 1:
                    decoder_inputs.append(emotion_embedding)
                if self.hparams.residual_encoder:
                    decoder_inputs.append(residual_encoding)

        decoder_input = torch.cat(decoder_inputs, -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        linear_inputs = [self.decoder_hidden, self.attention_context]
        if self.hparams.has_style_token_linear:
            if self.hparams.reference_encoder:
                linear_inputs.append(self.prosody_encoding)
            else:
                if len(self.hparams.all_speakers) > 1:
                    linear_inputs.append(speaker_embedding)
                if len(self.hparams.all_emotions) > 1:
                    linear_inputs.append(emotion_embedding)
                if self.hparams.residual_encoder:
                    linear_inputs.append(residual_encoding)
        decoder_hidden_attention_context = torch.cat(
            linear_inputs, dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights, self.attention_context, \
            self.prosody_encoding, prosody_pred

    def forward(self, memory, text_inputs, memory_lengths,
            speaker_embeddings, emotion_embeddings, residual_encoding,
            prosody_ref=None, global_prosody_ref=None,
            decoder_inputs=None, output_lengths=None, teacher_forcing=True,
            discrete_attention_weight=False,
            stop_prediction2=False):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        text_inputs: text inputs.
        - Type: torch.IntTensor.
        - Size: [B, max_text_length].
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        output_lengths: The length of Mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        prosody_ref: [B, seq_len, prosody_dim]

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        if teacher_forcing:
            decoder_input = self.get_go_frame(memory).unsqueeze(0)
            decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
            decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
            decoder_inputs = self.prenet(decoder_inputs)

            self.initialize_decoder_states(
                memory, mask=~get_mask_from_lengths(memory_lengths))
            if stop_prediction2:
                self.stop_predictor2.initialize(text_inputs, memory_lengths, output_lengths)

            mel_outputs, gate_outputs, alignments, attention_contexts, \
                prosody_encodings, prosody_preds = [], [], [], [], [], []
            while len(mel_outputs) < decoder_inputs.size(0) - 1:
                t = len(mel_outputs)
                decoder_input = decoder_inputs[t]
                t_prosody_ref = None
                if self.hparams.reference_encoder:
                    if self.hparams.reference_encoder == 'Glob2Temp':
                        if t == 0:
                            t_prosody_ref_hidden = global_prosody_ref
                        t_prosody_ref_hidden, t_prosody_ref = self.temp_prosody_decoder(None, t_prosody_ref_hidden)
                    else:
                        t_prosody_ref = prosody_ref[:,t,:]
                prev_global_prosody_ref = None
                if 'prev_global_prosody' in self.hparams.pp_opt_inputs:
                    if t == 0:
                        prev_global_prosody_ref = self.get_go_prosody(global_prosody_ref)
                    else:
                        prev_global_prosody_ref = global_prosody_ref[:,t-1,:]
                mel_output, gate_output, attention_weights, attention_context, \
                    prosody_encoding, prosody_pred = self.decode(
                    decoder_input, speaker_embeddings, emotion_embeddings,
                    residual_encoding, t_prosody_ref, prev_global_prosody_ref)
                mel_outputs += [mel_output.squeeze(1)]
                gate_outputs += [gate_output.squeeze(1)]
                alignments += [attention_weights]
                attention_contexts += [attention_context]
                prosody_encodings += [prosody_encoding]
                prosody_preds += [prosody_pred]

                if stop_prediction2:
                    _, end_points = self.stop_predictor2.predict(attention_weights)
                else:
                    end_points = None

            (mel_outputs, gate_outputs, alignments, attention_contexts,
                prosody_encodings, prosody_preds
                ) = self.parse_decoder_outputs(
                    mel_outputs, gate_outputs, alignments, attention_contexts,
                    prosody_encodings, prosody_preds)

            att_means = self.get_attention_means()

            return mel_outputs, gate_outputs, alignments, attention_contexts, \
                prosody_encodings, prosody_preds, end_points, att_means
        else:
            fr_outputs = self.free_running(memory, text_inputs, memory_lengths,
                speaker_embeddings, emotion_embeddings,
                residual_encoding, prosody_ref, global_prosody_ref,
                discrete_attention_weight, stop_prediction2)

            return fr_outputs

    def free_running(self, memory, text_inputs, memory_lengths, speaker_indices, emotion_vectors,
            residual_encoding, prosody_ref=None, global_prosody_ref=None, discrete_attention_weight=False,
            stop_prediction2=True):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        - Size: [batch_size, max_batch_input_len, enc_dim]
        - Type: torch.FloatTensor.
        text_inputs: Encoder inputs. Text int seqeunce.
        - Size: [batch_size, max_batch_input_len]
        - Type: torch.IntTensor.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder.
        - Size: [batch_size, n_mel_channels, max_decoder_steps]
        gate_outputs: gate outputs from the decoder
        - Size: [batch_size, max_decoder_steps, 1]
        alignments: sequence of attention weights from the decoder
        - Size: [batch_size, max_decoder_steps, batch_max_text_length]
        """
        prev_mel_output = self.get_go_frame(memory)

        #self.initialize_decoder_states(memory, mask=None)
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))
        if stop_prediction2:
            self.stop_predictor2.initialize(text_inputs, memory_lengths)
        if 'prev_global_prosody' in self.hparams.pp_opt_inputs:
            self.reference_encoder.initialize_states()

        end_decoding = False
        mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds \
            = [], [], [], [], [], []
        while (not end_decoding) and (len(mel_outputs) < self.max_decoder_steps):
            t = len(mel_outputs)
            t_temp_prosody_ref, prev_global_prosody_ref = None, None
            if 'prev_global_prosody' in self.hparams.pp_opt_inputs:
                t_temp_prosody_ref, prev_global_prosody_ref = self.reference_encoder(prev_mel_output.unsqueeze(-1))
                # squeeze time step dim
                t_temp_prosody_ref = t_temp_prosody_ref.squeeze(1)
                prev_global_prosody_ref = prev_global_prosody_ref.squeeze(1)
            elif self.hparams.reference_encoder == 'Glob2Temp':
                if t == 0:
                    t_temp_prosody_ref_hidden = global_prosody_ref
                t_temp_prosody_ref_hidden, t_temp_prosody_ref = self.temp_prosody_decoder(None, t_temp_prosody_ref_hidden)

            decoder_input = self.prenet(prev_mel_output)

            mel_output, gate_output, alignment, attention_context, \
                prosody_encoding, prosody_pred \
                = self.decode(decoder_input,
                speaker_indices, emotion_vectors, residual_encoding,
                t_prosody_ref=t_temp_prosody_ref,
                prev_global_prosody_ref=prev_global_prosody_ref,
                discrete_attention_weight=discrete_attention_weight)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [alignment]
            attention_contexts += [attention_context]
            prosody_encodings += [prosody_encoding]
            prosody_preds += [prosody_pred]
            prev_mel_output = mel_output

            if stop_prediction2:
                end_decoding, end_points = self.stop_predictor2.predict(alignment)
            else:
                end_decoding, end_points = False, None

        mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds \
            = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, attention_contexts,
            prosody_encodings, prosody_preds)

        return mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds, end_points

    def inference(self, memory, text_inputs, speaker_indices, emotion_vectors, residual_encoding,
            discrete_attention_weight=False):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        prev_mel_output = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)
        self.stop_predictor2.initialize(text_inputs, torch.IntTensor([text_inputs.size(-1)]))

        if 'prev_global_prosody' in self.hparams.pp_opt_inputs:
            self.reference_encoder.initialize_states()

        mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds \
            = [], [], [], [], [], []
        while True:
            t_temp_prosody_ref, prev_global_prosody_ref = None, None
            if 'prev_global_prosody' in self.hparams.pp_opt_inputs:
                t_temp_prosody_ref, prev_global_prosody_ref = self.reference_encoder(prev_mel_output.unsqueeze(-1))
                # squeeze time step dim
                t_temp_prosody_ref = t_temp_prosody_ref.squeeze(1)
                prev_global_prosody_ref = prev_global_prosody_ref.squeeze(1)
            decoder_input = self.prenet(prev_mel_output)
            mel_output, gate_output, alignment, attention_context, \
                prosody_encoding, prosody_pred \
                = self.decode(decoder_input,
                speaker_indices, emotion_vectors, residual_encoding,
                t_prosody_ref=t_temp_prosody_ref,
                prev_global_prosody_ref=prev_global_prosody_ref,
                discrete_attention_weight=discrete_attention_weight)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [alignment]
            attention_contexts += [attention_context]
            prosody_encodings += [prosody_encoding]
            prosody_preds += [prosody_pred]
            prev_mel_output = mel_output

            end_decoding, end_points = self.stop_predictor2.predict(alignment)
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                print("Stopped by 1st ending predictor")
                break
            if end_decoding:
                print("Stopped by 2nd ending predictor")
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments, attention_contexts,\
            prosody_encodings, prosody_preds \
            = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, attention_contexts,
            prosody_encodings, prosody_preds)

        return mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.hparams = hparams
        self.speaker_adversarial_training = hparams.speaker_adversarial_training
        self.emotion_adversarial_training = hparams.emotion_adversarial_training
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.int_dtype = torch.long
        self.float_dtype = torch.float32
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        if len(self.hparams.all_speakers) > 1:
            self.speaker_embedding_layer = SpeakerEncoder(hparams)
        if len(self.hparams.all_emotions) > 1:
            self.emotion_embedding_layer = EmotionEncoder(hparams)
        if hparams.speaker_adversarial_training:
            self.speaker_advgrad_classifier = SpeakerRevGradClassifier(hparams)
        else:
            self.speaker_advgrad_classifier = None
        if hparams.emotion_adversarial_training:
            self.emotion_advgrad_classifier = EmotionRevGradClassifier(hparams)
        else:
            self.emotion_advgrad_classifier = None
        if hparams.residual_encoder:
            self.residual_encoder = ResidualEncoder(hparams)
        else:
            self.residual_encoder = None
        if hparams.reference_encoder:
            if hparams.reference_encoder == 'LocalRefEncoder':
                self.reference_encoder = LocalRefEncoder(hparams)
            elif hparams.reference_encoder == 'GlocalRefEncoder':
                self.reference_encoder = GlocalRefEncoder(hparams)
                self.decoder.add_ref_enc(self.reference_encoder)
            elif hparams.reference_encoder == 'Glob2Temp':
                self.reference_encoder = ReferenceEncoder(hparams)
                self.temp_prosody_decoder = TemporalProsodyDecoder(hparams)
                self.decoder.add_temp_prosody_decoder(self.temp_prosody_decoder)
        else:
            self.reference_encoder = None

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, \
            speakers, sex, emotion_input_vectors, emotion_targets, lang, \
            text_raw, wav_paths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speakers = to_gpu(speakers).long()
        sex = to_gpu(sex).long()
        emotion_input_vectors = to_gpu(emotion_input_vectors).float()
        emotion_targets = to_gpu(emotion_targets).long()
        lang = to_gpu(lang).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
            (speakers, sex, emotion_input_vectors, emotion_targets, lang, \
                text_raw, wav_paths))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            # mask. dim0: batch_size. dim1:seq_len.
            mel_mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mel_mask = mel_mask.permute(1, 0, 2)

            # outputs[0]: mel_outputs
            outputs[0].data.masked_fill_(mel_mask, 0.0)
            # outputs[1]: mel_outputs_postnet
            outputs[1].data.masked_fill_(mel_mask, 0.0)
            # outputs[2]: gate_outputs
            outputs[2].data.masked_fill_(mel_mask[:, 0, :], 1e3)  # gate energies

            # outputs[4]: prosody_encodings
            if outputs[4] is not None:
                prosody_encodings_mask = mask.expand(outputs[4].size(2), mask.size(0), mask.size(1))
                prosody_encodings_mask = prosody_encodings_mask.permute(1, 2, 0)
                outputs[4].masked_fill_(prosody_ref_mask, 0.0)
            # outputs[5]: prosody_preds
            if outputs[5] is not None:
                prosody_preds_mask = mask.expand(outputs[5].size(2), mask.size(0), mask.size(1))
                prosody_preds_mask = prosody_preds_mask.permute(1, 2, 0)
                outputs[5].masked_fill_(prosody_preds_mask, 0.0)

        return outputs

    def forward(self, inputs, speakers, emotion_vectors,
            stop_prediction2=False,
            teacher_forcing=True, zero_res_en=False,
            discrete_attention_weight=False):
        self.int_dtype = speakers.dtype
        self.float_dtype = emotion_vectors.dtype
        if teacher_forcing:
            text_inputs, text_lengths, mels, max_len, output_lengths = inputs
            text_lengths, output_lengths = text_lengths.data, output_lengths.data

            embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

            encoder_outputs = self.encoder(embedded_inputs, text_lengths)

            if len(self.hparams.all_speakers) > 1:
                speaker_embeddings = self.speaker_embedding_layer(speakers)
            else:
                speaker_embeddings = None
            if len(self.hparams.all_emotions) > 1:
                emotion_embeddings = self.emotion_embedding_layer(emotion_vectors)
            else:
                emotion_embeddings = None

            if self.hparams.residual_encoder:
                if zero_res_en:
                    batch_size = text_inputs.size(0)
                    residual_encoding = self.residual_encoder.inference(batch_size, self.float_dtype)
                    mu = torch.zeros_like(residual_encoding)
                    logvar = torch.zeros_like(residual_encoding)
                else:
                    residual_encoding, mu, logvar = self.residual_encoder(mels)
            else:
                residual_encoding, mu, logvar = None, None, None

            if self.hparams.reference_encoder:
                if self.hparams.reference_encoder in ['GlocalRefEncoder']:
                    self.reference_encoder.initialize_states()
                prosody_ref, global_prosody_ref = self.reference_encoder(mels) # [batch_size, seq_len, prosody_dim]
            else:
                prosody_ref = None
                global_prosody_ref = None

            (mel_outputs, gate_outputs, alignments,
                attention_contexts,
                prosody_encodings, prosody_preds,
                end_points, att_means) = self.decoder(
                    encoder_outputs, text_inputs, text_lengths,
                        speaker_embeddings, emotion_embeddings, residual_encoding,
                        prosody_ref, global_prosody_ref,
                        mels, output_lengths, stop_prediction2=stop_prediction2, teacher_forcing=True)

            mel_outputs_postnet = self.postnet(mel_outputs)
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet

            if self.speaker_adversarial_training:
                spk_adv_batch = get_spk_adv_inputs(encoder_outputs, text_lengths)
                logit_speakers, prob_speakers, int_pred_speakers = self.speaker_advgrad_classifier(spk_adv_batch)
            else:
                logit_speakers, prob_speakers, int_pred_speakers = None, None, None

            if self.emotion_adversarial_training:
                emo_adv_batch = get_emo_adv_inputs(encoder_outputs, text_lengths)
                logit_emotions, prob_emotions, int_pred_emotions = self.emotion_advgrad_classifier(emo_adv_batch)
            else:
                logit_emotions, prob_emotions, int_pred_emotions = None, None, None

            outputs = self.parse_output([mel_outputs, mel_outputs_postnet,
                gate_outputs, alignments, prosody_encodings, prosody_preds, end_points],
                output_lengths)

            return (outputs,
                    (logit_speakers, prob_speakers, int_pred_speakers),
                    (logit_emotions, prob_emotions, int_pred_emotions),
                    (residual_encoding, mu, logvar),
                    att_means)
        else:

            fr_outputs = self.free_running(inputs, speakers, emotion_vectors,
                discrete_attention_weight, stop_prediction2)

            return fr_outputs

    def free_running(self, inputs, speakers, emotion_vectors,
            discrete_attention_weight=False, stop_prediction2=True):
        if self.hparams.reference_encoder == 'Glob2Temp':
            text_inputs, text_lengths, ref_mels, max_ref_mel_len, ref_mel_lengths = inputs
        else:
            text_inputs, text_lengths = inputs

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        if len(self.hparams.all_speakers) > 1:
            speaker_embeddings = self.speaker_embedding_layer(speakers)
        else:
            speaker_embeddings = None
        if len(self.hparams.all_emotions) > 1:
            emotion_embeddings = self.emotion_embedding_layer(emotion_vectors)
        else:
            emotion_embeddings = None
        if self.hparams.residual_encoder:
            batch_size = text_inputs.size(0)
            residual_encoding = self.residual_encoder.inference(batch_size, self.float_dtype)
        else:
            residual_encoding = None

        if self.hparams.reference_encoder == 'Glob2Temp':
            _, global_prosody_ref = self.reference_encoder(ref_mels, ref_mel_lengths)

        mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds, end_points \
            = self.decoder(
            encoder_outputs, text_inputs, text_lengths,
            speaker_embeddings, emotion_embeddings,
            residual_encoding,
            global_prosody_ref=global_prosody_ref,
            teacher_forcing=False,
            discrete_attention_weight=discrete_attention_weight,
            stop_prediction2=stop_prediction2)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output([mel_outputs, mel_outputs_postnet,
            gate_outputs, alignments, prosody_encodings, prosody_preds, end_points])

        return outputs

    def inference(self, text_inputs, speakers, emotion_vectors,
            discrete_attention_weight=False):
        if self.fp16_run:
            emotion_vectors = emotion_vectors.type(self.float_dtype)
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        if len(self.hparams.all_speakers) > 1:
            speaker_embeddings = self.speaker_embedding_layer(speakers)
        else:
            speaker_embeddings = None
        if len(self.hparams.all_emotions) > 1:
            emotion_embeddings = self.emotion_embedding_layer(emotion_vectors)
        else:
            emotion_embeddings = None
        if self.hparams.residual_encoder:
            batch_size = text_inputs.size(0)
            residual_encoding = self.residual_encoder.inference(batch_size, emotion_vectors.dtype)
        else:
            residual_encoding = None

        mel_outputs, gate_outputs, alignments, attention_contexts, \
            prosody_encodings, prosody_preds \
            = self.decoder.inference(
            encoder_outputs, text_inputs, speaker_embeddings, emotion_embeddings, residual_encoding,
            discrete_attention_weight)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output([
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments,
            prosody_encodings, prosody_preds])

        return outputs

    def get_embeddings(self, speakers, emotion_vectors):
        if len(self.hparams.all_speakers) > 1:
            speaker_embeddings = self.speaker_embedding_layer(speakers)
        else:
            speaker_embeddings = None
        if len(self.hparams.all_emotions) > 1:
            emotion_vectors = emotion_vectors.type(self.float_dtype)
            emotion_embeddings = self.emotion_embedding_layer(emotion_vectors)
        else:
            emotion_embeddings = None

        return speaker_embeddings, emotion_embeddings


class SpeakerEncoder(nn.Module):
    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()
        self.out_dim = hparams.speaker_embedding_dim
        self.max_speakers = len(hparams.all_speakers)
        #self.linear_projection = torch.nn.Linear(in_features=self.max_speakers, out_features=self.out_dim, bias=False)
        self.linear_projection = nn.Embedding(self.max_speakers, self.out_dim)

    def forward(self, inputs):
        speaker_embeddings = self.linear_projection(inputs)

        return speaker_embeddings


class EmotionEncoder(nn.Module):
    '''
    Emotion encoder should be constructed under #emotions > 1.
    '''
    def __init__(self, hparams):
        super(EmotionEncoder, self).__init__()
        self.out_dim = hparams.emotion_embedding_dim
        self.max_emotions = len(hparams.all_emotions)
        if hparams.neutral_zero_vector:
            self.linear_projection = torch.nn.Linear(in_features=self.max_emotions-1, out_features=self.out_dim, bias=False)
            #self.linear_projection = LinearNorm(self.max_emotions-1, self.out_dim, bias=False, w_init_gain='linear')
        else:
            self.linear_projection = torch.nn.Linear(in_features=self.max_emotions, out_features=self.out_dim, bias=False)
            #self.linear_projection = LinearNorm(self.max_emotions, self.out_dim, bias=False, w_init_gain='linear')


    def forward(self, inputs):
        emotion_embeddings = self.linear_projection(inputs)

        return emotion_embeddings


class LanguageEncoder(nn.Module):
    def __init__(self, hparams):
        super(LanguageEncoder, self).__init__()
        self.out_dim = hparams.lang_embedding_dim
        self.max_languages = hparams.max_languages
        #self.linear_projection = torch.nn.Linear(in_features=self.max_languages, out_features=self.out_dim, bias=False)
        self.linear_projection = nn.Embedding(self.max_languages, self.out_dim)

    def forward(self, inputs):
        language_embeddings = self.linear_projection(inputs)

        return language_embeddings


class SpeakerRevGradClassifier(nn.Module):
    def __init__(self, hparams):
        super(SpeakerRevGradClassifier, self).__init__()

        self.text_embedding_size = hparams.encoder_embedding_dim
        self.n_hidden_units = hparams.n_hidden_units
        self.max_speakers = len(hparams.all_speakers)
        self.revgrad_lambda = hparams.revgrad_lambda
        self.revgrad_max_grad_norm = hparams.revgrad_max_grad_norm

        self.linear_1 = torch.nn.Linear(in_features=self.text_embedding_size, out_features=self.n_hidden_units)
        self.linear_2 = torch.nn.Linear(in_features=self.n_hidden_units, out_features=self.max_speakers)

    def forward(self, inputs):
        revgrad_inputs = self.revgrad_layer(inputs, self.revgrad_lambda, self.revgrad_max_grad_norm)
        h = F.relu(self.linear_1(revgrad_inputs))
        logit_outputs = self.linear_2(h)
        prob_speakers = F.softmax(logit_outputs, dim=1)
        int_pred_speakers = torch.argmax(prob_speakers, dim=1)

        return logit_outputs, prob_speakers, int_pred_speakers

    # From https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/19
    def revgrad_layer(self, x, scale=1.0, max_grad_norm=0.5):
        GradientReverse.scale = scale
        GradientReverse.max_grad_norm = max_grad_norm
        return GradientReverse.apply(x)

class EmotionRevGradClassifier(nn.Module):
    def __init__(self, hparams):
        super(EmotionRevGradClassifier, self).__init__()

        self.text_embedding_size = hparams.encoder_embedding_dim
        self.n_hidden_units = hparams.n_hidden_units
        self.max_emotions = len(hparams.all_emotions)
        self.revgrad_lambda = hparams.revgrad_lambda
        self.revgrad_max_grad_norm = hparams.revgrad_max_grad_norm

        self.linear_1 = torch.nn.Linear(in_features=self.text_embedding_size, out_features=self.n_hidden_units)
        self.linear_2 = torch.nn.Linear(in_features=self.n_hidden_units, out_features=self.max_emotions)

    def forward(self, inputs):
        revgrad_inputs = self.revgrad_layer(inputs, self.revgrad_lambda, self.revgrad_max_grad_norm)
        h = F.relu(self.linear_1(revgrad_inputs))
        logit_outputs = self.linear_2(h)
        prob_emotions = F.softmax(logit_outputs, dim=1)
        int_pred_emotions = torch.argmax(prob_emotions, dim=1)

        return logit_outputs, prob_emotions, int_pred_emotions

    # From https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/19
    def revgrad_layer(self, x, scale=1.0, max_grad_norm=0.5):
        GradientReverse.scale = scale
        GradientReverse.max_grad_norm = max_grad_norm
        return GradientReverse.apply(x)

# From https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/19
class GradientReverse(torch.autograd.Function):
    scale = 1.0
    max_grad_norm = 0.5
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = GradientReverse.grad_clip(grad_output)
        return GradientReverse.scale * grad_output.neg()

    def grad_clip(grad_output):
        grad_norm = grad_output.norm().item()
        if grad_norm > GradientReverse.max_grad_norm:
            grad_output = grad_output / grad_norm * GradientReverse.max_grad_norm
        return grad_output


class ProsodyPredictorMLP(nn.Module):
    def __init__(self, hparams):
        super(ProsodyPredictorMLP, self).__init__()
        self.hparams = hparams
        in_dim = hparams.encoder_embedding_dim
        if len(self.hparams.all_speakers) > 1:
            in_dim += hparams.speaker_embedding_dim
        if len(hparams.all_emotions) > 1:
            in_dim += hparams.emotion_embedding_dim
        if 'prev_global_prosody' in hparams.pp_opt_inputs:
            if self.hparams.global_prosody_is_hidden:
                in_dim += hparams.ref_enc_gru_size
            else:
                in_dim += hparams.prosody_dim
        if 'AttRNN' in hparams.pp_opt_inputs:
            in_dim += hparams.attention_rnn_dim
        out_dim = hparams.prosody_dim
        self.linear1 = LinearNorm(in_dim, hparams.encoder_embedding_dim,
            bias=True, w_init_gain='relu')
        self.bn1 = nn.BatchNorm1d(hparams.encoder_embedding_dim)
        self.linear2 = LinearNorm(hparams.encoder_embedding_dim, hparams.encoder_embedding_dim,
            bias=True, w_init_gain='relu')
        self.bn2 = nn.BatchNorm1d(hparams.encoder_embedding_dim)
        self.linear3 = LinearNorm(hparams.encoder_embedding_dim, out_dim,
            bias=True, w_init_gain='relu')
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, inputs):

        h1 = F.relu(self.bn1(self.linear1(inputs)))
        h2 = F.relu(self.bn2(self.linear2(h1)))
        outputs = F.relu(self.bn3(self.linear3(h2)))

        return outputs


class ProsodyPredictorLSTMCell(nn.Module):
    def __init__(self, hparams):
        super(ProsodyPredictorLSTMCell, self).__init__()
        self.hparams = hparams
        self.pp_lstm_hidden_dim = hparams.pp_lstm_hidden_dim
        self.p_decoder_dropout = hparams.p_decoder_dropout
        in_dim = hparams.encoder_embedding_dim
        if len(self.hparams.all_speakers) > 1:
            in_dim += hparams.speaker_embedding_dim
        if len(hparams.all_emotions) > 1:
            in_dim += hparams.emotion_embedding_dim
        if 'prev_global_prosody' in hparams.pp_opt_inputs:
            if self.hparams.global_prosody_is_hidden:
                in_dim += hparams.ref_enc_gru_size
            else:
                in_dim += hparams.prosody_dim
        if 'AttRNN' in hparams.pp_opt_inputs:
            in_dim += hparams.attention_rnn_dim
        out_dim = hparams.prosody_dim
        self.cell = nn.LSTMCell(in_dim, self.pp_lstm_hidden_dim)
        self.linear = LinearNorm(self.pp_lstm_hidden_dim, out_dim,
            bias=True, w_init_gain='relu')
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, inputs, h_c):
        hidden, cell = self.cell(inputs, h_c)
        hidden = F.dropout(
            hidden, self.p_decoder_dropout, self.training)
        outputs = F.relu(self.bn(self.linear(hidden)))
        return outputs, hidden, cell


class TemporalProsodyDecoder(nn.Module):
    def __init__(self, hparams):
        super(TemporalProsodyDecoder, self).__init__()
        self.hparams = hparams
        self.prosody_dim = hparams.prosody_dim
        self.input_size = 1
        self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=hparams.ref_enc_gru_size)
        self.bn_lp = nn.BatchNorm1d(self.prosody_dim)
        self.linear_layer = LinearNorm(hparams.ref_enc_gru_size, self.prosody_dim, bias=False,
            w_init_gain='tanh')

    def forward(self, input, hidden):
        '''
        Params
        -----
        input:
        - size: [N, prosody_dim]

        Returns
        -----
        hidden:
        - size: [N, prosody_dim]
        '''
        if input is None:
            N = hidden.size(0)
            input = torch.zeros((N, self.input_size), dtype=hidden.dtype, requires_grad=False).cuda()
        hidden = self.gru_cell(input, hidden)
        t_prosody = F.tanh(self.bn_lp(self.linear_layer(hidden)))

        return hidden, t_prosody
