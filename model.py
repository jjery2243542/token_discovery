import torch 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from utils import cc 
from utils import gumbel_softmax

def _get_vgg2l_odim(idim, in_channel=1, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)
    return int(idim) * out_channel

def _pad_one_frame(inp):
    inp_t = inp.transpose(1, 2)
    out_t = F.pad(inp_t, (0, 1), mode='replicate')
    out = out_t.transpose(1, 2)
    return out

class VGG2L(torch.nn.Module):
    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        self.in_channel = in_channel
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

    def conv_block(self, inp, layers):
        out = inp
        for layer in layers:
            out = F.relu(layer(out))
        out = F.max_pool2d(out, 2, stride=2, ceil_mode=True)
        return out

    def forward(self, xs, ilens):
        # xs = [batch_size, frames, feeature_dim]
        # ilens is a list of frame length of each utterance 
        xs = torch.transpose(
                xs.view(xs.size(0), xs.size(1), self.in_channel, xs.size(2)//self.in_channel), 1, 2)
        xs = self.conv_block(xs, [self.conv1_1, self.conv1_2])
        xs = self.conv_block(xs, [self.conv2_1, self.conv2_2])
        ilens = np.array(np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64) 
        ilens = np.array(np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()
        xs = torch.transpose(xs, 1, 2)
        xs = xs.contiguous().view(xs.size(0), xs.size(1), xs.size(2) * xs.size(3))
        return xs, ilens

class pBLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate):
        super(pBLSTM, self).__init__()
        layers, dropout_layers = [], []
        for i in range(n_layers):
            idim = input_dim if i == 0 else hidden_dim * 2
            layers.append(torch.nn.LSTM(idim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True))
            dropout_layers.append(torch.nn.Dropout(p=dropout_rate, inplace=True))
        self.layers = torch.nn.ModuleList(layers)
        self.dropout_layers = torch.nn.ModuleList(dropout_layers)
        self.subsample = subsample

    def forward(self, xpad, ilens):
        for i, (layer, dropout_layer) in enumerate(zip(self.layers, self.dropout_layers)):
            # pack sequence 
            xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
            xs, (_, _) = layer(xpack)
            xpad, ilens = pad_packed_sequence(xs, batch_first=True)
            xpad = dropout_layer(xpad)
            ilens = ilens.numpy()
            # subsampling
            sub = self.subsample[i]
            if sub > 1:
                xpad = xpad[:, ::sub]
                ilens = [(length + 1) // sub for length in ilens]
        return xpad, ilens

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate, output_dim, in_channel=1):
        super(Encoder, self).__init__()
        self.enc1 = VGG2L(in_channel)
        out_channel = _get_vgg2l_odim(input_dim) 
        self.enc2 = pBLSTM(input_dim=out_channel, hidden_dim=hidden_dim, n_layers=n_layers, 
                subsample=subsample, dropout_rate=dropout_rate)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, ilens, temperature=1.0):
        out, ilens = self.enc1(x, ilens)
        out, ilens = self.enc2(out, ilens)
        out = self.linear(out)
        return out, ilens

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embedding_dim, n_latent):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Linear(n_latent, embedding_dim, bias=False)

    def forward(self, query, temperature=1.):
        '''
        query: batch_size x timestep x embedding_dim
        embedding: n_latent x embedding_dim
        '''
        unnormalized_logits = query @ self.embedding.weight
        unnormalized_logits = unnormalized_logits / torch.norm(query, p=2, dim=-1, keepdim=True)
        logits = unnormalized_logits / torch.norm(torch.t(self.embedding.weight), p=2, dim=-1)
        proba = gumbel_softmax(logits, temperature=temperature)
        output = self.embedding(proba)
        return proba, output

class AttLoc(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim, conv_channels, conv_kernel_size):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(encoder_dim, att_dim)
        self.mlp_dec = torch.nn.Linear(decoder_dim, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(conv_channels, att_dim, bias=False)
        if conv_kernel_size % 2 == 0:
            self.padding = (conv_kernel_size // 2, conv_kernel_size // 2 - 1)
        else:
            self.padding = (conv_kernel_size // 2, conv_kernel_size // 2)
        self.loc_conv = torch.nn.Conv2d(
                1, conv_channels, (1, conv_kernel_size), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.att_dim = att_dim
        self.conv_channels = conv_channels
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_pad, dec_z, att_prev, scaling=1.0):
        batch_size =enc_pad.size(0)
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_pad
            self.enc_length = self.enc_h.size(1)
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)
        
        dec_z = dec_z.view(batch_size, self.decoder_dim)

        # initialize attention weights
        if att_prev is None:
            att_prev = Variable(enc_pad.data.new(batch_size, self.enc_length).zero_())

        #att_prev: batch_size x frame
        att_prev_pad = F.pad(att_prev.view(batch_size, 1, 1, self.enc_length), self.padding)
        att_conv = self.loc_conv(att_prev_pad)
        # att_conv: batch_size x channel x 1 x frame -> batch_size x frame x channel
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: batch_size x frame x channel -> batch_size x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: batch_size x 1 x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch_size, 1, self.att_dim)
        att_state = torch.tanh(self.pre_compute_enc_h + dec_z_tiled + att_conv)
        e = self.gvec(att_state).squeeze(2)
        # w: batch_size x frame
        w = F.softmax(scaling * e, dim=1)
        # w_expanded: batch_size x 1 x frame
        w_expanded = w.unsqueeze(1)
        c = torch.bmm(w_expanded, self.enc_h).squeeze(1)
        return c, w 

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_dim, att_dim, hidden_dim, output_dim):
        # index 0 is padding, index 1 is GO symbol 
        self.input_layer = torch.nn.Linear(input_dim + 2, embedding_dim)
        self.rnn_cell = torch.nn.LSTMCell(embedding_dim + encoder_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.attention = AttLoc(encoder_dim=encoder_dim, decoder_dim=hidden_dim, 
                att_dim=att_dim, conv_channels=100, conv_kernel_size=10)

    def forward_step(self, token, last_hidden_state, encoder_state):
        self.rnn_cell()

if __name__ == '__main__':
    data = cc(torch.randn(32, 321, 13))
    ilens = np.ones((32,), dtype=np.int64) * 121
    net = cc(Encoder(13, 320, 4, [1, 2, 2, 1], dropout_rate=0.3, output_dim=512))
    emb = cc(EmbeddingLayer(embedding_dim=512, n_latent=300))
    out, ilens = net(data, ilens)
    print(out.size())
    distr, out = emb(out)
    print(distr.size(), out.size())
    #att = cc(AttLoc(640, 320, 300, 100, 10))
    #att.reset()
    #dec = cc(Variable(torch.randn(32, 320)))
    #context, weights = att(output, dec, None)
    #print(context.size(), weights.size(), weights[0])
    #dec = cc(Variable(torch.randn(32, 320)))
    #context, weights = att(output, dec, weights)
    #print(context.size(), weights.size(), weights[0])
    #dec = cc(Variable(torch.randn(32, 320)))
    #context, weights = att(output, dec, weights)
    #print(context.size(), weights.size(), weights[0])

