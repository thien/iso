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
    def __init__(self, embeddingMatrix, vocabularySize, padding_id, hiddenSize=512, bidirectional=False):
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
        self.gru = nn.GRU(embeddingDim, hiddenSize, bidirectional=self.isBidirectional)
    
    def forward(self, x, hidden):
        # load the input into the embedding before doing GRU computation.
        embed = self.embedding(x).view(1,x.size()[0],-1)
        output, hidden = self.gru(embed, hidden)
        return output, hidden
    
    def initHidden(self):
        numLayers = 1
        if self.isBidirectional:
            numLayers = 2 # because it's bidirectional
        return torch.zeros(numLayers, 73, self.hiddenSize, device=device)

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
        self.gru = nn.GRU(embeddingDim, hidden_size)

    def forward(self, x, hidden):
        embed = self.embedding(x).view(1,x.size()[0],-1)
        output, hidden = self.gru(embed, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 73, self.hiddenSize, device=device)

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
        print("ATTENTION:---------------------------")
        print("Encoder outputs shape:", encoderOutputs[0].shape)
        print("Prev Hidden shape:", prevHidden[0].shape)
        # concatenate hidden layer inputs together.
        concatenated = torch.cat((prevHidden[0], encoderOutputs[0]), 1)
        attentionWeights = F.softmax(self.attention(concatenated), dim=1)
        
        # batch matrix multiplication
        attentionApplied = torch.bmm(attentionWeights.unsqueeze(0),
                                    encoderOutputs[0].unsqueeze(0))

        context = torch.cat((prevHidden[0], attentionApplied[0]), 1)
        # reshape to produce context vector.
        context = self.attentionCombined(context).unsqueeze(0)
        context = F.relu(context)
        return context, attentionWeights

    def initHidden(self):
        return torch.zeros(1,1, self.hiddenSize) 

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class Decoder(nn.Module):
    def __init__(self, embeddingMatrix, vocabularySize, padding_id, outputSize, hiddenSize=512):
        """
        # dropout omitted
        """
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

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

        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)
        self.out = nn.Linear(self.hiddenSize * 2, vocabularySize)

        # dubious entry
        self.comb = nn.Linear(50,400)

    def forward(self, y, context, z, previousHidden):
        embedded = self.embedding(y).view(1,y.size()[0],-1) 
        print("DECODER------------------------------------")
        print("EMBED:",embedded[0].shape)
        print("CONTEXT:",context[0].shape)
        print("HIDDEN:", previousHidden.shape)
        print("Z:", z.shape)
   
        # this is diverging from the original paper but they're already
        # ambiguous on how the decoder is implemented
        # ("The input to GRU is the combination of the previous word's embedding
        # y_{t-1}, the context vector, ..., and the latent variable z.")
        # My guess is that we'll apply the attention mechanism 
        # and then do a linear combination of the result with z.

        inputs = torch.mm(torch.mm(self.comb(embedded[0]), z.transpose(0,1)), context)
        
        print("INPUTS SHAPE:", inputs.shape)
        # # concatenate hidden layer inputs together.
        # inputs = torch.cat((embedded[0], context, z), 1)



        # do a forward GRU
        output, hidden = self.gru(inputs.view(1,1,-1), previousHidden[0].view(1,1,-1))
        print("OUTPUT SHAPE:", output.shape)
        # softmax the output
        print("OUTS:", output[0,0].shape, context[0].shape)
        output = self.out(torch.cat((output[0,0], context[0]), 0))
        # output = F.log_softmax(output, dim=1)
        output = F.log_softmax(output, dim=0)
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
    def __init__(self, hidden_size=512, latent_size=400):
        super(Inference, self).__init__()
        
        # encode
        self.fc1  = nn.Linear(hidden_size + hidden_size + hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        # decode
        # self.fc3 = nn.Linear(latent_size + class_size, 400)
        # self.fc4 = nn.Linear(400, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
  

    # def decode(self, z, c): # P(x|z, c)
    #     '''
    #     z: (bs, latent_size)
    #     c: (bs, class_size)
    #     '''
    #     inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
    #     h3 = self.relu(self.fc3(inputs))
    #     return self.sigmoid(self.fc4(h3))

    def forward(self, h_forward, c, h_backward):
        print("INFERENCE:-----------")
        mu, logvar = self.encode(h_forward, c, h_backward)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
        
class Prior(nn.Module):
    def __init__(self, hidden_size=512, latent_size=400):
        super(Prior, self).__init__()
        
        # self.feature_size = feature_size
        # self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(hidden_size + hidden_size, hidden_size)
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
    LL = criterion(y_predicted.view(1,-1), y.view(1))
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
    inputLength = x.size(0)
    targetLength = y.size(0)
    
    # set default loss
    loss = 0
    
    # set up encoder computation
    encoderHidden  = encoder.initHidden()
    backwardHidden = backwards.initHidden()
    
    # set up encoder outputs
    encoderOutputs, encoderHidden = encoder(x, encoderHidden)
    # compute backwards outputs
    backwardOutput, backwardHidden = backwards(torch.flip(y, [0]), backwardHidden)

    # set up the variables for decoder computation
    decoderInput = torch.tensor([[word2id["<eos>"]]], dtype=torch.long, device=device)
    decoderHidden = encoderHidden[-1]
    decoderOutput = None
    

    print("ENCODER OUTPUTS:", encoderOutputs[:,1].shape)
    # Run through the decoder one step at a time. This seems to be common practice across
    # all of the seq2seq based implementations I've come across on GitHub.
    print("TARGET LENGTH",targetLength)
    for t in range(targetLength):
        # get the context vector c
        c, _ = attention(encoderHidden, encoderOutputs)
        # compute the inference layer
        print("ATTENTION DIM:",c.shape)
        z_infer, infMean, infLogvar = inference(decoderHidden, c[:,t], backwardOutput[:,t])
        # compute the prior layer
        z_prior, priMean, priLogvar = prior(decoderHidden, c[:,t])
        # compute the output of each decoder state
        DecoderOut = decoder(decoderInput, c[t], z_infer, decoderHidden)
        decoderOutput, decoderHidden = DecoderOut
        
        # calculate the loss
        loss += loss_function(decoderOutput, y[t], z_infer, z_prior, criterion)
        # feed this output to the next input
        decoderInput = y[t]
        print("NEXT ITERATION")
    
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
             word2id,
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

if __name__ == "__main__":
    print("Loading parameters..", end=" ")
    hiddenSize = 512
    featureSize = 512
    iterations = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
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
    
    
    max_length = len(dataset[0])
    vocabularySize = len(id2word)
    embeddingDim = weightMatrix.shape[1]
    embedding_shape = weightMatrix.shape
    paddingID = word2id['<pad>']

    print("Embedding shape:", embedding_shape)
    print("Padding ID:", paddingID)

    print("Initialising model components..", end=" ")
    modelEncoder   = Encoder(weightMatrix, vocabularySize, paddingID, hiddenSize).to(device)
    modelAttention = Attention(maxLength=max_length).to(device)
    modelBackwards = Backwards(weightMatrix, vocabularySize, paddingID, hiddenSize).to(device)
    modelInference = Inference(hidden_size=hiddenSize).to(device)
    modelPrior     = Prior(hidden_size=hiddenSize).to(device)
    modelDecoder   = Decoder(weightMatrix, vocabularySize, paddingID, max_length, hiddenSize).to(device)
    print("Done.")

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