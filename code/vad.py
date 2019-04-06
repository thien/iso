from __future__ import unicode_literals, print_function, division
from io import open
import random
import os
import bcolz
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm

from vad_utils import loss_function, plotBatchLoss, batchData, loadDataset, saveModels

class Encoder(nn.Module):
    """
    This'll be a bi-directional GRU.
    Utilises equation (1) in the paper.
    """
    def __init__(self, 
                embeddingMatrix, 
                vocabularySize, 
                padding_id, 
                hiddenSize=512, 
                bidirectional=True,
                xavier=False):

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

        return output, hidden

    def initHidden(self, batch_size):
        # initialises encoder hidden matrix
        numLayers = 2 if self.isBidirectional else 1
        return torch.zeros(numLayers, batch_size, self.hiddenSize)

class Backwards(nn.Module):
    def __init__(self, 
                embeddingMatrix, 
                vocabularySize, 
                padding_id, 
                hidden_size=512, 
                bidirectionalEncoder=False, xavier=False):

        super(Backwards, self).__init__()

        self.hiddenSize = hidden_size
        embeddingDim = embeddingMatrix.shape[1]

        self.embedding = nn.Embedding(
            num_embeddings=vocabularySize,
            embedding_dim=embeddingDim,
            padding_idx=padding_id,
            _weight=torch.Tensor(embeddingMatrix)
        )
    
        self.embedding.weight.requires_grad = False

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

        self.attnSize = self.hidden_size * 2 if self.bidirectionalEncoder else self.hidden_size
        self.attn = nn.Linear(self.attnSize, self.hidden_size)
        
        if xavier:
            init.xavier_uniform_(self.attn.weight)

    def forward(self, encoder_outputs, hidden):
        # https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
        # Create variable to store attention energies
        attn_energies = self.score(hidden, encoder_outputs)
        attention_weights = F.softmax(attn_energies, dim=1)
        c = torch.bmm(attention_weights, encoder_outputs).squeeze(1)
        return c
    
    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output).transpose(2, 1)
        hidden = hidden.unsqueeze(1)
        score = hidden.bmm(energy)
        return score

class Decoder(nn.Module):
    def __init__(self, 
                embeddingMatrix, 
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

        embeddingDim = embeddingMatrix.shape[1]
        
        self.embedding = nn.Embedding(
            num_embeddings=vocabularySize,
            embedding_dim=embeddingDim,
            padding_idx=padding_id,
            _weight=torch.Tensor(embeddingMatrix)
        )

        self.embedding.weight.requires_grad = False

        encoderDim = self.hiddenSize
        if encoderBidirectional:
            encoderDim *= 2

        gruInputSize = embeddingDim + encoderDim + self.latentSize
        self.gru = nn.GRU(gruInputSize, self.hiddenSize, batch_first=True)
        self.out = nn.Linear(self.hiddenSize + encoderDim, vocabularySize)
        
        if xavier:
            init.xavier_uniform_(self.gru.weight_hh_l0)
            init.xavier_uniform_(self.gru.weight_ih_l0)
            init.xavier_uniform_(self.out.weight)

    def forward(self, y, context, z, previousHidden):
        # get output word
        embedded = self.embedding(y).squeeze(1)
                # print(decoder.training)
        # combine inputs together
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
        if bidirectionalEncoder:
            forwardInput += hidden_size
        # encode
        self.fc1 = nn.Linear(forwardInput, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU()
        
    def encode(self, h_forward, c, h_backward):
        inputs = torch.cat([h_forward, c, h_backward], 1)
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
        if bidirectionalEncoder:
            forwardInput += hidden_size

        # encode
        self.fc1  = nn.Linear(forwardInput, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU()
        
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
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        vocab = self.bow(z)
        return vocab
   
def trainVAD(
             device,
             batch_num,
             num_batches,
             x,
             y,
             xLength,
             yLength,
             encoder,
             attention,
             backwards,
             inference,
             prior,
             decoder,
             cbow,
             encoderOpt,
             attentionOpt,
             backwardsOpt,
             inferenceOpt,
             priorOpt,
             decoderOpt,
             cbowOpt,
             word2id,
             criterion_r,
             criterion_bow,
             useBOW,
             gradientClip
            ):

    """
    Represents a whole sequence iteration trained on a batch of reviews.
    """

    # initialise gradients
    encoderOpt.zero_grad()
    attentionOpt.zero_grad()
    backwardsOpt.zero_grad()
    inferenceOpt.zero_grad()
    priorOpt.zero_grad()
    decoderOpt.zero_grad()
    if useBOW:
        cbowOpt.zero_grad()

    # set default loss
    loss = 0
    ll_loss = 0
    kl_loss = 0
    aux_loss = 0

    # initalise input and target lengths, and vocab size
    ySeqlength, batchSize = yLength[0], x.shape[0]
    vocabularySize = decoder.embedding.weight.shape[0]
    # set up encoder and backward hidden vectors
    encoderHidden = encoder.initHidden(batchSize).to(device)
    backwardHidden = backwards.initHidden(batchSize).to(device)

    # set up encoder outputs
    encoderOutputs, encoderHidden = encoder(x, encoderHidden, xLength)

    # compute backwards outputs
    backwardOutput, backwardHidden = backwards(torch.flip(
        y, [0, 1]), yLength, backwardHidden)

    # set up the variables for decoder computation
    decoderInput = torch.tensor([[word2id["<sos>"]]] * batchSize, dtype=torch.long, device=device)

    decoderHidden = encoderHidden[-1]
    decoderOutput = None

    # Run through the decoder one step at a time. This seems to be common practice across
    # all of the seq2seq based implementations I've come across on GitHub.
    for t in range(ySeqlength):

        # get the context vector c
        c = attention(encoderOutputs, decoderHidden)

        # compute the inference layer
        z_infer, infer_mu, infer_logvar = inference(decoderHidden, c, backwardOutput[:,t])
        # compute the prior layer
        z_prior, prior_mu, prior_logvar = prior(decoderHidden, c)

        # compute the output of each decoder state
        DecoderOut = decoder(decoderInput, c, z_infer, decoderHidden)

        # update variables
        decoderOutput, decoderHidden = DecoderOut

        # compute reference CBOW
        ref_bow = torch.FloatTensor(batchSize, vocabularySize).zero_().to(device)
        ref_bow.scatter_(1,y[:,t:],1)

        # compute auxillary
        pred_bow = cbow(z_infer)

        # calculate the loss
        seqloss, ll, kl, aux = loss_function(
                                            batch_num, 
                                            num_batches,
                                            decoderOutput,
                                            y[:, t],
                                            infer_mu,
                                            infer_logvar,
                                            prior_mu,
                                            prior_logvar,
                                            ref_bow,
                                            pred_bow,
                                            criterion_r,
                                            criterion_bow)
        
        loss += seqloss
        ll_loss += ll
        kl_loss += kl
        aux_loss += aux
        
        # feed this output to the next input
        decoderInput = y[:,t]
        decoderHidden = decoderHidden.squeeze(0)
   
    # calculate gradients
    loss.backward()
    
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(backwards.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(attention.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(inference.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(prior.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradientClip)
    if useBOW:
        torch.nn.utils.clip_grad_norm_(cbow.parameters(), gradientClip)
    
    # gradient descent
    if useBOW:
        cbowOpt.step()
    decoderOpt.step()
    priorOpt.step()
    inferenceOpt.step()
    attentionOpt.step()
    backwardsOpt.step()
    encoderOpt.step()
    
    return loss.item()/ySeqlength, ll_loss.item()/ySeqlength, kl_loss.item()/ySeqlength, aux_loss.item()/ySeqlength

def trainIteration(
                device,
                dataset,
                encoder,
                attention,
                backwards,
                inference,
                prior,
                decoder,
                cbow,
                iterations,
                word2id,
                criterion_r,
                criterion_bow,
                learningRate,
                gradientClip,
                useBOW,
                printEvery = 10,
                plotEvery = 100):
    
    plotLosses = []
    printLossTotal = 0
    plotLossTotal = 0
    
    encoderOpt   = optim.Adam(encoder.parameters(),   lr=learningRate)
    attentionOpt = optim.Adam(attention.parameters(), lr=learningRate)
    backwardsOpt = optim.Adam(backwards.parameters(), lr=learningRate)
    inferenceOpt = optim.Adam(inference.parameters(), lr=learningRate)
    priorOpt     = optim.Adam(prior.parameters(),     lr=learningRate)
    decoderOpt   = optim.Adam(decoder.parameters(),   lr=learningRate)
    cbowOpt      = optim.Adam(cbow.parameters(),      lr=learningRate)
    
    numBatches = len(dataset[0])

    for j in range(1, iterations + 1):
        print("Iteration", j)
        # set up variables needed for training.
        n = -1
        
        # get random indexes
        indexes = [i for i in range(numBatches)]
        random.shuffle(indexes)
        
        # note that the data entries in each batch are sorted!
        # we're shuffling the batches.

        losses = []
        ll_losses = []
        kl_losses = []
        aux_losses = []
        for batch_num in tqdm(indexes):
            n += 1
            # each batch is composed of the 
            # reviews, and a sentence length.
            x, xLength = dataset[0][batch_num][0], dataset[0][batch_num][1]
            y, yLength = dataset[1][batch_num][0], dataset[1][batch_num][1]

            # calculate loss.
            loss, ll, kl, aux = trainVAD(
                device,
                n,
                numBatches,
                x, y, 
                xLength,  
                yLength,
                encoder,
                attention,
                backwards,
                inference,
                prior,
                decoder,
                cbow,
                encoderOpt,
                attentionOpt,
                backwardsOpt,
                inferenceOpt,
                priorOpt,
                decoderOpt,
                cbowOpt,
                word2id,
                criterion_r,
                criterion_bow,
                useBOW,
                gradientClip
                )
            
            print("Batch:",n,"Loss:",round(loss,4), "LL:", round(ll,4), "KL:", round(kl,4), "AUX:", round(aux,4))

            # increment our print and plot.
            printLossTotal += loss
            plotLossTotal += loss
            
            losses.append(loss)
            ll_losses.append(ll)
            kl_losses.append(kl)
            aux_losses.append(aux)
            
            if batch_num % 10 == 0:
                plotBatchLoss(j, ll_losses, kl_losses, aux_losses)
        
        saveModels(encoder, backwards, attention, inference, prior, decoder, cbow)
