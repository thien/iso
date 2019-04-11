from __future__ import unicode_literals, print_function, division
from io import open
import random
import os
import bcolz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from vad_utils import loss_function, plotBatchLoss, batchData, loadDataset, saveModels, sequence_mask, masked_cross_entropy



class Encoder(nn.Module):
    """
    This'll be a bi-directional GRU.
    Utilises equation (1) in the paper.
    """
    def __init__(self, 
                embedding, 
                vocabularySize, 
                padding_id, 
                hiddenSize=512, 
                bidirectional=True,
                xavier=False):

        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.isBidirectional = bidirectional

        embeddingDim = embedding.weight.shape[1]
        self.embedding = embedding

        self.gru = nn.GRU(embeddingDim, self.hiddenSize,
                          bidirectional=self.isBidirectional, batch_first=True)
        
        if xavier:
            init.xavier_uniform_(self.gru.weight_hh_l0)
            init.xavier_uniform_(self.gru.weight_hh_l0_reverse)
            init.xavier_uniform_(self.gru.weight_ih_l0)
            init.xavier_uniform_(self.gru.weight_ih_l0_reverse)
    
    def forward(self, x, hidden, x_length=None):
        # load the input into the embedding before doing GRU computation.
        embed = self.embedding(x)

        if x_length is not None:
            # load pack_padded_sequence so PyTorch knows when to not compute rubbish
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed, x_length, batch_first=True)
        
        # load it through the GRU
        packed_outputs, hidden = self.gru(packed_emb, hidden)
    
        if x_length is not None:
            # reverse pack_padded_sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)


        if self.isBidirectional:
            output = output[:, :, :self.hiddenSize] + output[:, : ,self.hiddenSize:]
        
        return output, hidden

    def initHidden(self, batch_size):
        # initialises encoder hidden matrix
        numLayers = 2 if self.isBidirectional else 1
        return torch.zeros(numLayers, batch_size, self.hiddenSize)

class Backwards(nn.Module):
    def __init__(self, 
                embedding, 
                vocabularySize, 
                padding_id, 
                hidden_size=512, 
                bidirectionalEncoder=False, xavier=False):

        super(Backwards, self).__init__()

        self.hiddenSize = hidden_size
        embeddingDim = embedding.weight.shape[1]

        self.embedding = embedding

        self.numLayers = 2 if bidirectionalEncoder else 1
        self.gru = nn.GRU(embeddingDim, 
                          hidden_size,
                          num_layers=self.numLayers,
                          batch_first=True)

        if xavier:
            init.xavier_uniform_(self.gru.weight_hh_l0)
            init.xavier_uniform_(self.gru.weight_ih_l0)
        
    def forward(self, x, x_length, hidden):
        # load the input into the embedding before doing GRU computation.
        embed = self.embedding(x)

        if x_length is not None:
            # load pack_padded_sequence so PyTorch knows when to not compute rubbish
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed, x_length, batch_first=True)

        # load it through the GRU
        output, hidden = self.gru(packed_emb, hidden)

        if x_length is not None:
            # reverse pack_padded_sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden

    def initHidden(self, batchSize):
        return torch.zeros(self.numLayers, batchSize, self.hiddenSize)

class Attention(nn.Module):
    def __init__(self,  hidden_size, bidirectionalEncoder, xavier=False):
        super(Attention, self).__init__()
        self.bidirectionalEncoder = bidirectionalEncoder
        self.hidden_size = hidden_size

        self.attnSize = self.hidden_size

        self.attn = nn.Linear(self.hidden_size, hidden_size)

        self.W_c = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        
        if xavier:
            init.xavier_uniform_(self.attn.weight)
            init.xavier_uniform_(self.W_c.weight)

    def forward(self, encoder_outputs, src_lens, hidden, device):

        # attention_scores: (batch_size, seq_len=1, max_src_len)
        energy = self.attn(encoder_outputs).transpose(2, 1)
        hidden = hidden.unsqueeze(1)
        attention_scores = hidden.bmm(energy)

        # attention_mask: (batch_size, seq_len=1, max_src_len)
        attention_mask = sequence_mask(src_lens).unsqueeze(1).to(device)

        # Fills elements of tensor with `-float('inf')` where `mask` is 1.
        attention_scores.data.masked_fill_(1 - attention_mask.data, -float('inf'))

        # attention_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax` 
        # => (batch_size, seq_len=1, max_src_len)
        attention_weights = F.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)

        # context_vector:
        # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
        # => (batch_size, seq_len=1, encoder_hidden_size * num_directions)
        # print("ATTN WEIGHTS:",attention_weights.shape)
        context_vector = torch.bmm(attention_weights, encoder_outputs)

        # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
        concat_input = torch.cat([context_vector, hidden], -1)

        # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size, seq_len=1, decoder_hidden_size)
        concat_output = torch.tanh(self.W_c(concat_input)).squeeze(1)
        
        # Prepare returns:
        # (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len)
        # attention_weights = attention_weights.squeeze(1)
        return concat_output

class Decoder(nn.Module):
    def __init__(self, 
                embedding, 
                vocabularySize, 
                padding_id, 
                batchSize, 
                outputSize, 
                hiddenSize, 
                latentSize, 
                encoderBidirectional,
                xavier=False):

        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.latentSize = latentSize

        embeddingDim = embedding.weight.shape[1]

        # additional components
        self.attention = Attention(hiddenSize, encoderBidirectional)
        self.inference = Inference(hiddenSize, latentSize, encoderBidirectional)
        self.prior = Prior(hiddenSize, latentSize, encoderBidirectional)
        self.cbow = CBOW(vocabularySize, latentSize)

        # decoder components:
        self.embedding = embedding

        encoderDim = self.hiddenSize

        gruInputSize = embeddingDim + encoderDim + self.latentSize
   
        self.gru = nn.GRU(gruInputSize, self.hiddenSize, batch_first=True)
        self.out = nn.Linear(self.hiddenSize + encoderDim, vocabularySize)
        
        if xavier:
            init.xavier_uniform_(self.gru.weight_hh_l0)
            init.xavier_uniform_(self.gru.weight_ih_l0)
            init.xavier_uniform_(self.out.weight)

    def forward(self, y, encoderOutputs, encoderLengths, previousHidden, device, back=None):

        # CALCULATE ATTENTION ---------------------------------
        c = self.attention(encoderOutputs, encoderLengths, previousHidden, device)

        # LATENT SAMPLING -------------------------------------

        if self.training:
            # compute the inference layer
            z, infer_mu, infer_logvar = self.inference(
                previousHidden, c, back)
            # compute the prior layer
            _, prior_mu, prior_logvar = self.prior(previousHidden, c)

        else:
            # otherwise just sample from the prior layer.
            z, _, _ = self.prior(previousHidden, c)

        # AUXILIARY FUNCTION ----------------------------------

        if self.training:
            sbow = self.cbow(z)

        # DECODER COMPONENT -----------------------------------

        # get output word
        embedded = self.embedding(y).squeeze(1)

        # combine inputs together
        inputs = torch.cat([embedded,c,z], 1).unsqueeze(1)

        if len(previousHidden.shape) < 3:
            previousHidden = previousHidden.unsqueeze(0)

        # do a forward GRU
        output, hidden = self.gru(inputs, previousHidden)

        # softmax the output
        output = output.squeeze(1)
        output = torch.cat((output, c), 1)
        output = self.out(output)
    
        if self.training:
            return output, hidden, sbow, infer_mu, infer_logvar, prior_mu, prior_logvar
        else:
            return output, hidden

    def initHidden(self):
        return torch.zeros(self.batchSize, 1, self.hiddenSize) 

class Inference(nn.Module):
    """
    Note that the inference and prior networks
    are a simple 1 layer feed forward neural network.
    Therefore the size of the weights are entirely based on the size
    of the input and outputs.
    """
    def __init__(self, hidden_size=512, latent_size=400, bidirectionalEncoder=False):
        super(Inference, self).__init__()
        
        forwardInput = hidden_size * 3
        # encode
        self.fc1 = nn.Linear(forwardInput, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.mean.weight)
        init.xavier_uniform_(self.var.weight)
        
    def encode(self, h_forward, c, h_backward):
        inputs = torch.cat([h_forward, c, h_backward], 1)
        # print(inputs.shape)
        h1 = self.relu(self.fc1(inputs))
        z_mu = self.mean(h1)
        z_var = self.var(h1)
        return z_mu, z_var
    
    def reparametrize(self, mu, logvar):
        # samples your mu, logvar to get z.
        std = torch.exp(0.5 * logvar)
        epsilon = logvar.new_empty(logvar.size()).normal_()
        return mu + std * epsilon
  
    def forward(self, h_forward, c, h_backward):
        mu, logvar = self.encode(h_forward, c, h_backward)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
        
class Prior(nn.Module):
    def __init__(self, hidden_size=512, latent_size=400, bidirectionalEncoder=False):
        super(Prior, self).__init__()
        
        forwardInput = hidden_size * 2

        # encode
        self.fc1  = nn.Linear(forwardInput, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var  = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.mean.weight)
        init.xavier_uniform_(self.var.weight)
        
    def encode(self, h, c):
        inputs = torch.cat([h, c], 1)
        h1 = self.relu(self.fc1(inputs))
        z_mu = self.mean(h1)
        z_var = self.var(h1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = logvar.new_empty(logvar.size()).normal_()
        return mu + std * epsilon

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

class CBOW(nn.Module):
    def __init__(self, vocabulary_size, latent_size=400):
        super(CBOW, self).__init__()
        
        self.bow = nn.Linear(latent_size, vocabulary_size)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, z):
        # vocab = -F.log_softmax(self.bow(z), dim=1)
        vocab = -self.sigmoid(self.bow(z))
        return vocab
