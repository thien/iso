from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import bcolz
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math
import pickle
from tqdm import tqdm
from torch.autograd import Variable

"""
Set seed #
"""
seed = 1337

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def loadDataset(path = '../Datasets/Reviews/dataset_ready.pkl'):
    return pickle.load(open(path, 'rb'))

class Encoder(nn.Module):
    """
    This'll be a bi-directional GRU.
    Utilises equation (1) in the paper.
    
    The hidden size is 512 as per the paper.
    """
    def __init__(self, embeddingMatrix, vocabularySize, padding_id, hiddenSize=512, bidirectional=True):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.isBidirectional = bidirectional
        # this embedding is a simple lookup table that stores the embeddings of a 
        # fixed dictionary and size.
        
        # This module is often used to store word embeddings and retrieve them
        # using indices. 
        # The input to the module is a list of indices, and 
        # the output is the corresponding word embeddings.
        embeddingDim = embeddingMatrix.shape[1]
        self.embedding = nn.Embedding(
            num_embeddings=vocabularySize,
            embedding_dim=embeddingDim,
            padding_idx=padding_id,
            _weight=torch.Tensor(embeddingMatrix)
        )
        # we're using pretrained labels
        self.embedding.weight.requires_grad = False
        self.embedding.to(device)
        self.gru = nn.GRU(embeddingDim, hiddenSize,
                          bidirectional=self.isBidirectional, batch_first=True)
    
    def forward(self, x, x_length, hidden):
        # load the input into the embedding before doing GRU computation.
        embed = self.embedding(x)
        # load pack_padded_sequence so PyTorch knows when to not compute rubbish
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embed, x_length, batch_first=True)
        # load it through the GRU
        packed_outputs, hidden = self.gru(packed_emb, hidden)
        # reverse
        output, output_lens = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        if self.isBidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)
        return output, hidden
    
    def initHidden(self, batch_size):
        numLayers = 1
        if self.isBidirectional:
            numLayers = 2 # because it's bidirectional
        return torch.zeros(numLayers, batch_size, self.hiddenSize, device=device)

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden

class Backwards(nn.Module):
    def __init__(self, embeddingMatrix, vocabularySize, padding_id, hidden_size=512):
        super(Backwards, self).__init__()
        self.hiddenSize = hidden_size

        embeddingDim = embeddingMatrix.shape[1]

        self.embedding = nn.Embedding(
            num_embeddings=vocabularySize,
            embedding_dim=embeddingDim,
            padding_idx=padding_id,
            _weight=torch.Tensor(embeddingMatrix)
        )
        # we're using pretrained labels
        self.embedding.weight.requires_grad = False
        self.embedding.to(device)
        self.gru = nn.GRU(embeddingDim, hidden_size, batch_first=True)

    def forward(self, x, x_length, hidden):
        embed = self.embedding(x)
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embed, x_length, batch_first=True)
        packed_outputs, hidden = self.gru(packed_emb, hidden)
        output, output_lens = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True)
        return output, hidden

    def initHidden(self, batchSize):
        return torch.zeros(1, batchSize, self.hiddenSize, device=device)

class Attn(nn.Module):
    def __init__(self,  hidden_size):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).squeeze()
    
    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = energy.transpose(2, 1)
        hidden = hidden.unsqueeze(1)
        energy = hidden.bmm(energy)
        return energy

class Decoder(nn.Module):
    def __init__(self, embeddingMatrix, vocabularySize, padding_id, batchSize, outputSize, hiddenSize, latentSize, encoderBidirectional):
        """
        # dropout omitted
        """
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.latentSize = latentSize

        embeddingDim = embeddingMatrix.shape[1]
        self.embedding = nn.Embedding(
            num_embeddings=vocabularySize,
            embedding_dim=embeddingDim,
            padding_idx=padding_id,
            _weight=torch.Tensor(embeddingMatrix)
        )
        # we're using pretrained labels
        self.embedding.weight.requires_grad = False
        self.embedding.to(device)

        encoderDim = self.hiddenSize
        if encoderBidirectional:
            encoderDim *= 2

        self.gru = nn.GRU(encoderDim +
                          embeddingDim + self.latentSize, encoderDim, batch_first=True)
        self.out = nn.Linear(encoderDim * 2, vocabularySize)

    def forward(self, y, context, z, previousHidden):
        embedded = self.embedding(y).squeeze(1)

        inputs = torch.cat([embedded,context,z], 1).unsqueeze(1)

        if len(previousHidden.shape) < 3:
            previousHidden = previousHidden.unsqueeze(0)
        # do a forward GRU
        output, hidden = self.gru(inputs, previousHidden)
        # softmax the output
        output = output.squeeze(1)
        output = torch.cat((output, context), 1)
        output = self.out(output)

        output = F.log_softmax(output, dim=0)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.batchSize, 1, self.hiddenSize, device=device) 

class Inference(nn.Module):
    """
    Note that the inference and prior networks
    are a simple 1 layer feed forward neural network.
    Therefore the size of the weights are entirely based on the size
    of the input and outputs.
    """
    def __init__(self, hidden_size=512, latent_size=400):
        super(Inference, self).__init__()
        
        # encode
        self.fc1  = nn.Linear(hidden_size*2 + hidden_size*2 + hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        self.relu = nn.ReLU()

    def encode(self, h_forward, c, h_backward): # Q(z|x, c)
        inputs = torch.cat([h_forward, c, h_backward], 1) # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        z_mu = self.mean(h1)
        z_var = self.var(h1)
        return z_mu, z_var
    
    def reparametrize(self, mu, logvar):
        # samples your mu, logvar to get z.
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu
  
    def forward(self, h_forward, c, h_backward):
        # print("INFERENCE:-----------")
        mu, logvar = self.encode(h_forward, c, h_backward)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
        
class Prior(nn.Module):
    def __init__(self, hidden_size=512, latent_size=400):
        super(Prior, self).__init__()
        
        # self.feature_size = feature_size
        # self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(hidden_size*2 + hidden_size*2, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        self.relu = nn.ReLU()

    def encode(self, h, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([h, c], 1) # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        z_mu = self.mean(h1)
        z_var = self.var(h1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        # samples your mu, logvar to get z.
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(y_predicted, y, z_inference, z_prior, criterion):
    LL = criterion(y_predicted, y)
    KL = F.kl_div(z_inference, z_prior)
    return LL - KL

def trainVAD(x,
             y,
             xLength,
             yLength,
             encoder,
             attention,
             backwards,
             inference,
             prior,
             decoder,
             encoderOpt,
             attentionOpt,
             backwardsOpt,
             inferenceOpt,
             priorOpt,
             decoderOpt,
             word2id,
             criterion = nn.NLLLoss()
            ):

    """
    Represents a whole sequence iteration trained on one review.
    """
    
    # initialise gradients
    encoderOpt.zero_grad()
    attentionOpt.zero_grad()
    backwardsOpt.zero_grad()
    inferenceOpt.zero_grad()
    priorOpt.zero_grad()
    decoderOpt.zero_grad()
    
    # initalise input and target lengths
    inputLength = x[0].size(0)
    targetLength = y[0].size(0)
    batchSize = x.shape[0]

    # set default loss
    loss = 0
    
    # set up encoder computation
    encoderHidden = encoder.initHidden(batchSize)
    backwardHidden = backwards.initHidden(batchSize)
    
    # set up encoder outputs
    encoderOutputs, encoderHidden = encoder(x, xLength, encoderHidden)
    # compute backwards outputs
    backwardOutput, backwardHidden = backwards(torch.flip(
        y, [0, 1]), yLength, backwardHidden)


    # set up the variables for decoder computation
    decoderInput = torch.tensor([[word2id["<eos>"]]] * batchSize, dtype=torch.long, device=device)
    
    decoderHidden = encoderHidden[-1]
    decoderOutput = None
    
    # Run through the decoder one step at a time. This seems to be common practice across
    # all of the seq2seq based implementations I've come across on GitHub.
    for t in range(targetLength-1):
        # print("ITERATION:-----------------------------------")
        # get the context vector c
        # c, _ = attention(encoderOutputs, decoderHidden)
        # print("DECODER HIDDEN:--------", decoderHidden.shape)
        c = attention(decoderHidden, encoderOutputs)
        # compute the inference layer
        # print("ATTENTION DIM:",c.shape)

        why = encoderOutputs
        c = c.unsqueeze(1)
        # print("C HOW:", c.shape)
        # print("INP:", why.shape)
        what = torch.bmm(c, why).squeeze(1)
        # print(what.shape)

        z_infer, _, _ = inference(decoderHidden, what, backwardOutput[:,t])
        # compute the prior layer
        z_prior, _, _ = prior(decoderHidden, what)
        # compute the output of each decoder state
        DecoderOut = decoder(decoderInput, what, z_infer, decoderHidden)
        decoderOutput, decoderHidden = DecoderOut
        # calculate the loss
        loss += loss_function(decoderOutput, y[:,t], z_infer, z_prior, criterion)
        # feed this output to the next input
        decoderInput = y[:,t]
        decoderHidden = decoderHidden.squeeze(0)
    
    # possible because our loss_function uses gradient storing calculations
    loss.backward()
    
    encoderOpt.step()
    attentionOpt.step()
    backwardsOpt.step()
    inferenceOpt.step()
    priorOpt.step()
    decoderOpt.step()
    
    return loss.item()/targetLength

def trainIteration(
                dataset,
                encoder,
                attention,
                backwards,
                inference,
                prior,
                decoder,
                iterations,
                word2id,
                criterion = nn.NLLLoss(),
                learningRate = 0.0001,
                printEvery = 10,
                plotEvery = 100):
    
    start = time.time()
    plotLosses = []
    printLossTotal = 0
    plotLossTotal = 0
    
    encoderOpt   = optim.Adam(encoder.parameters(),   lr=learningRate)
    attentionOpt = optim.Adam(attention.parameters(), lr=learningRate)
    backwardsOpt = optim.Adam(backwards.parameters(), lr=learningRate)
    inferenceOpt = optim.Adam(inference.parameters(), lr=learningRate)
    priorOpt     = optim.Adam(prior.parameters(),     lr=learningRate)
    decoderOpt   = optim.Adam(decoder.parameters(),   lr=learningRate)
    
    for j in range(1, iterations + 1):
        print("Iteration", j)
        # set up variables needed for training.
        for i in range(1,len(dataset)+1):
            batch = dataset[i-1]
            # each batch is composed of the 
            # reviews, and a sentence length.
            entry, length = batch
            x, y = entry, entry
            xLength, yLength = length, length

            # calculate loss.
            loss = trainVAD(x, y, 
                xLength,  
                yLength,
                encoder,
                attention,
                backwards,
                inference,
                prior,
                decoder,
                encoderOpt,
                attentionOpt,
                backwardsOpt,
                inferenceOpt,
                priorOpt,
                decoderOpt,
                word2id,
                criterion
                )
            # increment our print and plot.
            printLossTotal += loss
            plotLossTotal += loss
            
            # print mechanism
            print(loss)
            # plot mechanism
            # if i % plotEvery == 0:
            #     plotLossAvg = plotLossTotal / plotEvery
            #     plotLosses.append(plotLossAvg)
            #     plotLossTotal = 0


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

"""
Dataset batching mechanism
"""
def batchData(dataset, eos, batchsize=32):
    """
    Splits the dataset into batches.
    Each batch needs to be sorted by 
    the length of their sequence in order
    for `pack_padded_sequence` to be used.
    """
    datasize = len(dataset)
    batches = []
    # split data into batches.
    for i in range(0, datasize, batchsize):
        batches.append(dataset[i:i+batchsize])
    # within each batch, sort the entries.
    for i in range(len(batches)):
        batch = batches[i]
        # get lengths of each review in the batch
        # based on the postion of the EOS tag.
        lengths = (batch==eos).nonzero()[:,1]
        # sort the lengths
        ordered = torch.argsort(lengths, descending=True)
        # get the reviews based on the sorted batch lengths
        reviews = batch[ordered]
        # re-allocate values.
        batches[i] = (reviews, lengths[ordered])
    return batches


if __name__ == "__main__":
    print("Loading parameters..", end=" ")
    hiddenSize = 128
    featureSize = 128
    latentSize = 128
    iterations = 10
    bidirectionalEncoder = True
    batchSize = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Done.")

    print("Loading dataset..", end=" ")
    dataset = loadDataset()
    # setup store parameters
    id2word = dataset['id2word']
    word2id = dataset['word2id']
    weightMatrix = dataset['weights']
    dataset = dataset['reviews']
    print("Done.")

    print("Converting dataset weights into tensors..", end=" ")
    # convert dataset into tensors
    dataset = torch.tensor(dataset, dtype=torch.long, device=device)
    weightMatrix = torch.tensor(weightMatrix, dtype=torch.float)
    print("Done.")

    # batching data
    print("Batching Data..",end=" ")
    dataset = batchData(dataset, word2id['<eos>'], batchSize)
    print("Done.")
    
    # setup variables for model components initialisation
    maxReviewLength = dataset[0][0].shape[1]
    vocabularySize = len(id2word)
    embeddingDim = weightMatrix.shape[1]
    embedding_shape = weightMatrix.shape
    paddingID = word2id['<pad>']

    print("Initialising model components..", end=" ")
    modelEncoder = Encoder(weightMatrix, vocabularySize,
                           paddingID, hiddenSize, bidirectionalEncoder).to(device)
    # modelAttention = Attention(maxLength=maxReviewLength).to(device)
    modelAttention = Attn(hiddenSize).to(device)
    modelBackwards = Backwards(weightMatrix, vocabularySize, paddingID, hiddenSize).to(device)
    modelInference = Inference(hiddenSize, latentSize).to(device)
    modelPrior = Prior(hiddenSize, latentSize).to(device)
    modelDecoder = Decoder(weightMatrix, vocabularySize,
                           paddingID, batchSize, maxReviewLength, hiddenSize, latentSize, bidirectionalEncoder).to(device)
    print("Done.")

    print()
    trainIteration(dataset,
                   modelEncoder,
                   modelAttention,
                   modelBackwards,
                   modelInference,
                   modelPrior,
                   modelDecoder,
                   iterations,
                   word2id, 
                   printEvery=1000)
