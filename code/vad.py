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


"""
Set seed #
"""
seed = 1337

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def loadDataset(path = '../Datasets/Reviews/dataset_ready.pkl'):
    print("Loading Reviews..", end=" ")
    reviews = pickle.load(open(path, 'rb'))
    print(" Done.")
    return reviews

class Encoder(nn.Module):
    """
    This'll be a bi-directional GRU.
    Utilises equation (1) in the paper.
    
    The hidden size is 512 as per the paper.
    """
    def __init__(self, embeddingMatrix, inputSize, padding_id, hiddenSize=512):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        # this embedding is a simple lookup table that stores the embeddings of a 
        # fixed dictionary and size.
        
        # This module is often used to store word embeddings and retrieve them
        # using indices. 
        # The input to the module is a list of indices, and 
        # the output is the corresponding word embeddings.
        embeddingDim = len(embeddingMatrix[1])
        self.embedding = nn.Embedding(
            num_embeddings=inputSize,
            embedding_dim=embeddingDim,
            padding_idx=padding_id,
            _weight=torch.Tensor(embeddingMatrix)
        )
        # we're using pretrained labels
        self.embedding.weight.requires_grad = False
        self.embedding.to(device)
        self.gru = nn.GRU(embeddingDim, hiddenSize, bidirectional=True)
    
    def forward(self, x, hidden):
        # load the input into the embedding before doing GRU computation.
        embed = self.embedding(x).view(1,1,-1)
        # print("EMBED DIM:", embed.size())
        # print("HIDDEN DIM:", hidden.size())
        output, hidden = self.gru(embed, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(2,1, self.hiddenSize, device=device)

class Backwards(nn.Module):
    def __init__(self, embeddingMatrix, inputSize, padding_id, hidden_size=512):
        super(Backwards, self).__init__()
        self.hiddenSize = hidden_size

        embeddingDim = len(embeddingMatrix[1])

        self.embedding = nn.Embedding(
            num_embeddings=inputSize,
            embedding_dim=embeddingDim,
            padding_idx=padding_id,
            _weight=torch.Tensor(embeddingMatrix)
        )
        # we're using pretrained labels
        self.embedding.weight.requires_grad = False
        self.embedding.to(device)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize, device=device)

class Attention(nn.Module):
    """
    TODO: Add layer normalisation?
    https://arxiv.org/abs/1607.06450
    
    We also set the hidden state size to 512.
    
    """
    def __init__(self, maxLength, hiddenSize=512):
        """
        # dropout omitted
        """
        super(Attention, self).__init__()
        self.hiddenSize = hiddenSize
        self.maxLength = maxLength
        
        # self.attention is our tiny neural network that takes 
        # in the hidden weights and the previous hidden weights.
        self.attention = nn.Linear(self.hiddenSize * 2, self.maxLength)
        self.attentionCombined = nn.Linear(self.hiddenSize * 2, self.hiddenSize)
        # torch.nn.init.xavier_uniform_(self.attention)
        # torch.nn.init.xavier_uniform_(self.attentionCombined)
    
    def forward(self, prevHidden, encoderOutputs):

        # concatenate hidden layer inputs together.
        attentionInputs  = prevHidden
        attentionWeights = F.softmax(self.attention(attentionInputs), dim=1)
        
        # batch matrix multiplication
        attentionApplied = torch.bmm(attentionWeights.unsqueeze(0),
                                    encoderOutputs.unsqueeze(0))
        # reshape to produce context vector.
        context = self.attentionCombined(context).unsqueeze(0)
        context = F.relu(context)
        return context, attentionWeights

    def initHidden(self):
        return torch.zeros(1,1, self.hiddenSize) 

class Decoder(nn.Module):
    def __init__(self, outputSize, hiddenSize=512):
        """
        # dropout omitted
        """
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)
        self.out = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, previousY, previousHidden, context, z):
        # concatenate hidden layer inputs together.
        inputs = torch.cat((embedded[0], context, z), 1)
        # do a forward GRU
        output, hidden = self.gru(context, previousHidden)
        # softmax the output
        output = self.out(torch.cat((output[0], context), 1))
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1, self.hiddenSize, device=device) 

class Inference(nn.Module):
    """
    Note that the inference and prior networks
    are a simple 1 layer feed forward neural network.
    Therefore the size of the weights are entirely based on the size
    of the input and outputs.
    """
    def __init__(self, input_size, h_size, hidden_size=512, latent_size=400):
        super(Inference, self).__init__()
        
        # encode
        self.fc1  = nn.Linear(input_size + h_size + h_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        # decode
        # self.fc3 = nn.Linear(latent_size + class_size, 400)
        # self.fc4 = nn.Linear(400, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, h_forward, c, h_backward): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
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
  

    # def decode(self, z, c): # P(x|z, c)
    #     '''
    #     z: (bs, latent_size)
    #     c: (bs, class_size)
    #     '''
    #     inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
    #     h3 = self.relu(self.fc3(inputs))
    #     return self.sigmoid(self.fc4(h3))

    def forward(self, h_forward, c, h_backward):
        mu, logvar = self.encode(h_forward, c, h_backward)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
        
class Prior(nn.Module):
    def __init__(self, input_size, h_size, hidden_size=512, latent_size=400):
        super(Prior, self).__init__()
        
        # self.feature_size = feature_size
        # self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(input_size + h_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        # # decode
        # self.fc3 = nn.Linear(latent_size + class_size, 400)
        # self.fc4 = nn.Linear(400, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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

    # def decode(self, z, c): # P(x|z, c)
    #     '''
    #     z: (bs, latent_size)
    #     c: (bs, class_size)
    #     '''
    #     inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
    #     h3 = self.relu(self.fc3(inputs))
    #     return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
        # return self.decode(z, c), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(y_predicted, y, z_inference, z_prior, criterion):
    LL = criterion(y_predicted, y)
    KL = F.kl_div(z_inference, z_prior)
    return LL - KL

def trainVAD(x,
             y,
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
             maxLength,
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
    inputLength = x.size(0)
    targetLength = y.size(0)
    
    # set default loss
    loss = 0
    
    # set up encoder computation
    encoderOutput  = torch.zeros(maxLength, encoder.hiddenSize, device=device)
    encoderHidden  = encoder.initHidden()
    backwardOutput = torch.zeros(maxLength, backwards.hiddenSize, device=device)
    backwardHidden = torch.zeros(maxLength, backwards.hiddenSize, device=device)
    decoderHidden  = torch.zeros(maxLength, decoder.hiddenSize, device=device)
    
    # set up encoder outputs
    for ei in range(inputLength):
        encoderOutput, encoderHidden = encoder(x[ei], encoderHidden)
        encoderOutputs[ei] = encoderOutput[0,0]
    
    # set up backwards RNN
    for t in range(targetLength-1, 0, 1):
        # here we can also build the backwards RNN that takes in the y.
        # this backwards RNN conditions our latent variable.
        backwardOutput, backwardsHidden = backwards(y[t+1], backwardsHidden)
        # get the values of our backwards network.
        backwardOutputs[t] = backwardOutput[0,0]
        
    # set up the decoder computation
    decoderInput = torch.tensor([[SOS_token]], device=device)
    decoderHidden = encoderHidden
    
    for t in range(targetLength):
        # get the context vector c
        c, _ = attention(decoderH, encoderOutputs[t])
        # compute the inference layer
        z_infer, infMean, infLogvar = inference(decoderOutput, c, backwardOutputs[t])
        # compute the prior layer
        z_prior, priMean, priLogvar = prior(decoderOutput, c)
        # compute the output of each decoder state
        DecoderOut = decoder(decoderInput, c, z_infer, decoderHidden)
        decoderOutput, decoderHidden = DecoderOut
        
        # calculate the loss
        loss += loss_function(decoderOutput, y[t], z_infer, z_prior, criterion)
        # feed this output to the next input
        decoderInput = y[t]
    
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
                criterion = nn.NLLLoss(),
                learningRate = 0.0001,
                printEvery = 1000,
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
    
    for i in range(1, iterations + 1):
        # set up variables needed for training.
        entry = random.choice(dataset)
        x, y = entry, entry

        # calculate loss.
        loss = trainVAD(x, y, 
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
             criterion
            )
        # increment our print and plot.
        printLossTotal += loss
        plotLossTotal += loss
        
        # print mechanism
        if i % printEvery == 0:
            printLossAvg = printLossTotal / printEvery
            # reset the print loss.
            printLossTotal = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / iterations),
                                         i, i / iterations * 100, printLossAvg))
        # plot mechanism
        if i % plotEvery == 0:
            plotLossAvg = plotLossTotal / plotEvery
            plotLosses.append(plotLossAvg)
            plotLossTotal = 0
            
    showPlot(plotLosses)


if __name__ == "__main__":
    hiddenSize = 512
    featureSize = 512
    iterations = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = loadDataset()
    # setup store parameters
    id2word = dataset['id2word']
    word2id = dataset['word2id']
    weightMatrix = dataset['weights']
    dataset = dataset['reviews']

    # convert dataset into tensors
    dataset = torch.tensor(dataset, dtype=torch.long, device=device)

    max_length = len(dataset[0])
    vocabularySize = len(id2word)
    embeddingDim = len(weightMatrix[1])
    embedding_shape = weightMatrix.shape
    print("Embedding shape:", embedding_shape)

    paddingID = word2id['<pad>']
    print("Padding ID:", paddingID)

    modelEncoder   = Encoder(weightMatrix, vocabularySize, paddingID, hiddenSize).to(device)
    modelAttention = Attention(maxLength=max_length).to(device)
    modelBackwards = Backwards(weightMatrix, vocabularySize, paddingID, hiddenSize).to(device)
    modelInference = Inference(embedding_shape[0], embedding_shape[1], hiddenSize).to(device)
    modelPrior     = Prior(embedding_shape[0], embedding_shape[1], hiddenSize).to(device)
    modelDecoder   = Decoder(max_length, hiddenSize).to(device)

    trainIteration(dataset,
                   modelEncoder,
                   modelAttention,
                   modelBackwards,
                   modelInference,
                   modelPrior,
                   modelDecoder,
                   iterations,
                   75000, 
                   printEvery=1000)