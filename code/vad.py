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
    def __init__(self, 
                embeddingMatrix, 
                vocabularySize, 
                padding_id, 
                hiddenSize=512, 
                bidirectional=True):

        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.isBidirectional = bidirectional

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

        return output, hidden

    def initHidden(self, batch_size):
        numLayers = 2 if self.isBidirectional else 1
        return torch.zeros(numLayers, batch_size, self.hiddenSize, device=device)

class Backwards(nn.Module):
    def __init__(self, embeddingMatrix, vocabularySize, padding_id, hidden_size=512, bidirectionalEncoder=False):
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

        self.numLayers = 2 if bidirectionalEncoder else 1
        self.gru = nn.GRU(embeddingDim, hidden_size,num_layers=self.numLayers,batch_first=True)

    def forward(self, x, x_length, hidden):
        embed = self.embedding(x)
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embed, x_length, batch_first=True)
        packed_outputs, hidden = self.gru(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True)
        return output, hidden

    def initHidden(self, batchSize):
        return torch.zeros(self.numLayers, batchSize, self.hiddenSize, device=device)

class Attn(nn.Module):
    def __init__(self,  hidden_size, bidirectionalEncoder):
        super(Attn, self).__init__()
        self.bidirectionalEncoder = bidirectionalEncoder
        self.hidden_size = hidden_size

        self.attnInput = self.hidden_size * 2 if self.bidirectionalEncoder else self.hidden_size
        self.attn = nn.Linear(self.attnInput, self.hidden_size)
        # self.compressor = nn.Linear(self.attnInput, self.hidden_size)

    def forward(self, encoder_outputs, hidden):
        # https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
        # Create variable to store attention energies
        attn_energies = self.score(hidden, encoder_outputs)
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        attention_weights = F.softmax(attn_energies, dim=1)
        # compute context weights
        c = torch.bmm(attention_weights, encoder_outputs).squeeze(1)
    
        return c
    
    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output).transpose(2, 1)
        hidden = hidden.unsqueeze(1)
        score = hidden.bmm(energy)
        return score

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

        self.gru = nn.GRU(embeddingDim +
                          encoderDim + self.latentSize, self.hiddenSize, batch_first=True)
        self.out = nn.Linear(self.hiddenSize + encoderDim, vocabularySize)

    def forward(self, y, context, z, previousHidden):
        # print("Y:", y.shape)
        # print("C:", context.shape)
        # print("Z:", z.shape)
        # print("H", previousHidden.shape)
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
        output = F.log_softmax(output, dim=1)
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
    def __init__(self, hidden_size=512, latent_size=400, bidirectionalEncoder=False):
        super(Inference, self).__init__()
        
        forwardInput = hidden_size * 3
        if bidirectionalEncoder:
            forwardInput += hiddenSize
        # encode
        self.fc1 = nn.Linear(forwardInput, hidden_size)
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
        # return eps.mul(std) + mu
        return mu + std.mul(eps)
  
    def forward(self, h_forward, c, h_backward):
        mu, logvar = self.encode(h_forward, c, h_backward)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
        
class Prior(nn.Module):
    def __init__(self, hidden_size=512, latent_size=400, bidirectionalEncoder=False):
        super(Prior, self).__init__()
        
        forwardInput = hidden_size * 2
        if bidirectionalEncoder:
            forwardInput += hiddenSize

        # encode
        self.fc1  = nn.Linear(forwardInput, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        self.relu = nn.ReLU()

    def encode(self, h, c): # Q(z|x, c)
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
            return mu + std.mul(eps)
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

    # print("X SHAPE:", x.shape)
    # print("Y SHAPE:", y.shape)
    # set default loss
    loss = 0
    
    # set up encoder computation
    encoderHidden = encoder.initHidden(batchSize)
    backwardHidden = backwards.initHidden(batchSize)
    
    # set up encoder outputs
    encoderOutputs, encoderHidden = encoder(x, encoderHidden, xLength)

    # print("ENCODER OUTPUT:", encoderOutputs.shape)
    # print("ENCODER HIDDEN:", encoderHidden.shape)

    # compute backwards outputs
    backwardOutput, backwardHidden = backwards(torch.flip(
        y, [0, 1]), yLength, backwardHidden)

    # print("BACKWARD OUTPUT:", backwardOutput.shape)
    # print("BACKWARD HIDDEN:", backwardHidden.shape)

    # set up the variables for decoder computation
    decoderInput = torch.tensor([[word2id["<eos>"]]] * batchSize, dtype=torch.long, device=device)
    
    decoderHidden = encoderHidden[-1]
    decoderOutput = None
    
    # Run through the decoder one step at a time. This seems to be common practice across
    # all of the seq2seq based implementations I've come across on GitHub.
    for t in range(targetLength-1):
        # get the context vector c
        c = attention(encoderOutputs, decoderHidden)

        # compute the inference layer
        z_infer, _, _ = inference(decoderHidden, c, backwardOutput[:,t])
        # compute the prior layer
        z_prior, _, _ = prior(decoderHidden, c)
        # print("DECODER:")
        # compute the output of each decoder state
        DecoderOut = decoder(decoderInput, c, z_infer, decoderHidden)
        # update variables
        decoderOutput, decoderHidden = DecoderOut
        # calculate the loss
        seqloss = loss_function(decoderOutput, y[:,t], z_infer, z_prior, criterion)
        loss += seqloss

        # feed this output to the next input
        decoderInput = y[:,t]
        decoderHidden = decoderHidden.squeeze(0)
    print(loss)
    # possible because our loss_function uses gradient storing calculations
    loss.backward()
    print(loss)
    decoderOpt.step()
    priorOpt.step()
    inferenceOpt.step()
    attentionOpt.step()
    backwardsOpt.step()
    encoderOpt.step()
    
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
        n = -1
        for batch in dataset:
            n += 1
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
            
            print("BATCH ",n,"- LOSS:", loss)
            if n > 0:
                break
        break


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
    hiddenSize = 64
    latentSize = 32
    batchSize  = 16
    iterations = 1
    learningRate = 0.01
    bidirectionalEncoder = True
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
    modelAttention = Attn(hiddenSize, bidirectionalEncoder).to(device)
    modelBackwards = Backwards(weightMatrix, vocabularySize,
                               paddingID, hiddenSize, bidirectionalEncoder).to(device)
    modelInference = Inference(
        hiddenSize, latentSize, bidirectionalEncoder).to(device)
    modelPrior = Prior(hiddenSize, latentSize, bidirectionalEncoder).to(device)
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
                   learningRate=learningRate,
                   printEvery=1000)
