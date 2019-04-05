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
import torch.nn.init as init
import time
import math
import pickle
from tqdm import tqdm
from torch.autograd import Variable

# plotting
import matplotlib
matplotlib.use('Agg')

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
#         self.embedding.to(device)    
        
        self.gru = nn.GRU(embeddingDim, self.hiddenSize,
                          bidirectional=self.isBidirectional, batch_first=True)
        
#         init.xavier_uniform_(self.gru.bias_hh_l0)
#         init.xavier_uniform_(self.gru.bias_hh_l0_reverse)
#         init.xavier_uniform_(self.gru.bias_ih_l0)
#         init.xavier_uniform_(self.gru.bias_ih_l0_reverse)
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
        numLayers = 2 if self.isBidirectional else 1
        return torch.zeros(numLayers, batch_size, self.hiddenSize)

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
#         self.embedding.to(device)

        self.numLayers = 2 if bidirectionalEncoder else 1
        self.gru = nn.GRU(embeddingDim, hidden_size,num_layers=self.numLayers,batch_first=True)

        init.xavier_uniform_(self.gru.weight_hh_l0)
        init.xavier_uniform_(self.gru.weight_ih_l0)
        
    def forward(self, x, x_length, hidden):
        embed = self.embedding(x)
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embed, x_length, batch_first=True)
        packed_outputs, hidden = self.gru(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True)
        return output, hidden

    def initHidden(self, batchSize):
        return torch.zeros(self.numLayers, batchSize, self.hiddenSize)

class Attn(nn.Module):
    def __init__(self,  hidden_size, bidirectionalEncoder):
        super(Attn, self).__init__()
        self.bidirectionalEncoder = bidirectionalEncoder
        self.hidden_size = hidden_size

        self.attnInput = self.hidden_size * 2 if self.bidirectionalEncoder else self.hidden_size
        self.attn = nn.Linear(self.attnInput, self.hidden_size)
        
        init.xavier_uniform_(self.attn.weight)

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
#         self.embedding.to(device)

        encoderDim = self.hiddenSize
        if encoderBidirectional:
            encoderDim *= 2

        self.gru = nn.GRU(embeddingDim +
                          encoderDim + self.latentSize, self.hiddenSize, batch_first=True)
        self.out = nn.Linear(self.hiddenSize + encoderDim, vocabularySize)
        
        init.xavier_uniform_(self.gru.weight_hh_l0)
        init.xavier_uniform_(self.gru.weight_ih_l0)
        init.xavier_uniform_(self.out.weight)

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
        
#         init.kaiming_uniform_(self.fc1.weight)
#         init.kaiming_uniform_(self.mean.weight)
#         init.kaiming_uniform_(self.var.weight)

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
            forwardInput += hidden_size

        # encode
        self.fc1  = nn.Linear(forwardInput, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        self.relu = nn.ReLU()
        
#         init.kaiming_uniform_(self.fc1.weight)
#         init.kaiming_uniform_(self.mean.weight)
#         init.kaiming_uniform_(self.var.weight)

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

class CBOW(nn.Module):
    def __init__(self, vocabulary_size, latent_size=400):
        super(CBOW, self).__init__()
        
        self.bow = nn.Linear(latent_size, vocabulary_size)
        
        init.xavier_uniform_(self.bow.weight)
        
        self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
    
    def forward(self, z):
#         vocab = self.sigmoid(self.bow(z))
        vocab = self.bow(z)
#         onehot = self.softmax(vocab)
        
        return vocab
        
def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    mu_1, var_1 = recog_mu, recog_logvar
    mu_2, var_2 = prior_mu, prior_logvar
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    kld = -0.5 * torch.sum(1 + (var_1 - var_2)
                           - torch.div(torch.pow(mu_2 - mu_1,
                                                 2), torch.exp(var_2))
                           - torch.div(torch.exp(var_1), torch.exp(var_2)), 1)
    return kld
    
def loss_function(batch_num,
                  num_batches, 
                  y_predicted, 
                  y, 
                  inference_mu,
                  inference_logvar, 
                  prior_mu, 
                  prior_logvar, 
                  ref_bow, 
                  pred_bow,
                  criterion_r, 
                  criterion_bow):
    
    # compute reconstruction loss
    LL = criterion_r(y_predicted, y)
    
    # compute KLD
    KL = gaussian_kld(inference_mu, inference_logvar, prior_mu, prior_logvar)
    KL = torch.mean(torch.mean(KL))
    
    # KL Annealing
    kl_weight = 0 if batch_num == 0 else batch_num/num_batches
    weighted_KL = KL * kl_weight 
    
    # compute auxillary loss
    aux = criterion_bow(pred_bow, ref_bow)
    # weight auxillary loss
    alpha = 10
    weighted_aux = aux * alpha
    
    return LL + weighted_KL + weighted_aux, LL, weighted_KL, weighted_aux

def trainVAD(
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
             criterion_reconstruction,
             criterion_bow,
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
    cbowOpt.zero_grad()

    # set default loss
    loss = 0
    ll_loss = 0
    kl_loss = 0
    aux_loss = 0

    # initalise input and target lengths
    inputLength, targetLength, batchSize = x[0].size(0), y[0].size(0), x.shape[0]
    ySeqlength = yLength[0]

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
        seqloss, ll, kl, aux = loss_function(batch_num, num_batches, decoderOutput, y[:, t], infer_mu, infer_logvar, prior_mu, prior_logvar, ref_bow, pred_bow, criterion_reconstruction, criterion_bow)
        
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
    torch.nn.utils.clip_grad_norm_(cbow.parameters(), gradientClip)
    
    # gradient descent
    cbowOpt.step()
    decoderOpt.step()
    priorOpt.step()
    inferenceOpt.step()
    attentionOpt.step()
    backwardsOpt.step()
    encoderOpt.step()
    
    return loss.item()/targetLength, ll_loss.item()/targetLength, kl_loss.item()/targetLength, aux_loss.item()/targetLength

def trainIteration(
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
                criterion_reconstruction,
                criterion_bow,
                learningRate,
                gradientClip,
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
                criterion_reconstruction,
                criterion_bow,
                gradientClip
                )
            
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

def plotBatchLoss(iteration, losses, kl, aux):
    x = [i for i in range(1,len(losses)+1)]
    
#     su = plt.plot(x, losses)
#     losses = losses - kl - aux
    labels = ["KL", "Auxiliary", "LL"]

    plt.stackplot(x, kl, aux, losses, labels=labels)
    plt.legend()
    title = 'Learning Loss during Iteration ' + str(iteration)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Batch Number')
    
    filetype = "png"
    directory = "charts"
    filename = title + "." + filetype
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath, bbox_inches='tight')    
    plt.close()
        
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

def padSeq(row, maxlength, padID, cutoff):
    currentLength = len(row)
    difference = maxlength - currentLength
    return row + [padID for _ in range(difference)]


def batchData(dataset, padID, device, batchsize=32, cutoff=50):
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
        lengths = [len(seq) for seq in batch]
        indexes = [x for x in range(len(lengths))]
        sortedindexes = sorted(list(zip(lengths, indexes)), reverse=True)

        # since sentences are split by period, the period itself acts
        # the token to identify that the sentence has ended.
        # i.e. we don't need another token identifying the end of the subsequence.

        # get the reviews based on the sorted batch lengths
        reviews = [padSeq(batch[i[1]], cutoff, padID, cutoff)
                   for i in sortedindexes]

        reviews = torch.tensor(reviews, dtype=torch.long, device=device)
        # re-allocate values.
        batches[i] = (reviews, [i[0] for i in sortedindexes])
    return batches


def saveModels(encoder, backwards, attention, inference, prior, decoder, cbow):
    print("Saving models..", end=" ")
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(attention.state_dict(), 'attention.pth')
    torch.save(backwards.state_dict(), 'backwards.pth')
    torch.save(inference.state_dict(), 'inference.pth')
    torch.save(prior.state_dict(), 'prior.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')
    torch.save(cbow.state_dict(), 'cbow.pth')
    print("Done.")
    
    
if __name__ == "__main__":
    print("Loading parameters..", end=" ")
    hiddenSize = 512
    latentSize = 400
    batchSize  = 32
    iterations = 2
    learningRate = 0.0001
    gradientClip = 1
    bidirectionalEncoder = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = "cpu"
    print("Done.")

    print("Loading dataset..", end=" ")
    dataset = loadDataset()
    # setup store parameters
    id2word = dataset['id2word']
    word2id = dataset['word2id']
    weightMatrix = dataset['weights']
    train = dataset['train']
    validation = dataset['validation']
    cutoff = dataset['cutoff']
    paddingID = word2id['<pad>']
    print("Done.")

    print("Converting dataset weights into tensors..", end=" ")
    # convert dataset into tensors
    weightMatrix = torch.tensor(weightMatrix, dtype=torch.float)
    print("Done.")

    # batching data
    print("Batching Data..",end=" ")
    
    # shuffle data rows
    random.shuffle(train)
    random.shuffle(validation)

    trainx = [x[0] for x in train]
    trainy = [x[1] for x in train]
#     valx = [x[0] for x in validation]
#     valy = [x[1] for x in validation]

    # shuffle data row

    trainx = batchData(trainx, paddingID, device, batchSize, cutoff)
    trainy = batchData(trainy, paddingID, device, batchSize, cutoff)
    # trainx = batchData(valx, paddingID, batchSize, cutoff)
    # trainy = batchData(valy, paddingID, batchSize, cutoff)

    traindata = (trainx, trainy)
    # valdata= (valx, valy)
    print("Done.")
    
    # setup variables for model components initialisation
    maxReviewLength = cutoff
    vocabularySize = len(id2word)
    embeddingDim = weightMatrix.shape[1]
    embedding_shape = weightMatrix.shape


    print("Initialising model components..", end=" ")

    modelEncoder = Encoder(weightMatrix, vocabularySize,
                           paddingID, hiddenSize, bidirectionalEncoder).to(device)

    modelAttention = Attn(hiddenSize, bidirectionalEncoder).to(device)
    modelBackwards = Backwards(weightMatrix, vocabularySize,
                               paddingID, hiddenSize, bidirectionalEncoder).to(device)
    modelInference = Inference(
        hiddenSize, latentSize, bidirectionalEncoder).to(device)
    modelPrior = Prior(hiddenSize, latentSize, bidirectionalEncoder).to(device)
    modelDecoder = Decoder(weightMatrix, vocabularySize,
                           paddingID, batchSize, maxReviewLength, hiddenSize, latentSize, bidirectionalEncoder).to(device)
    modelBOW = CBOW(vocabularySize, latentSize).to(device)
    
    criterion_r = nn.NLLLoss(ignore_index=paddingID)
    criterion_bow = nn.BCEWithLogitsLoss()
    
    print("Done.")

    trainIteration(traindata,
                   modelEncoder,
                   modelAttention,
                   modelBackwards,
                   modelInference,
                   modelPrior,
                   modelDecoder,
                   modelBOW,
                   iterations,
                   word2id, 
                   criterion_r,
                   criterion_bow,
                   learningRate,
                   gradientClip,
                   printEvery=1000)

    saveModels(modelEncoder, modelBackwards, modelAttention, modelInference, modelPrior, modelDecoder, modelBOW)
 