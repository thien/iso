from vad import Encoder, Backwards, Attention, Decoder, Inference, Prior, CBOW
from vad_utils import loss_function, plotBatchLoss, batchData, loadDataset


import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

seed = 1337

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def trainVAD(
    device,
    batch_num,
    num_batches,
    x,
    y,
    xLength,
    yLength,
    encoder,
    backwards,
    decoder,
    encoderOpt,
    backwardsOpt,
    decoderOpt,
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
    backwardsOpt.zero_grad()
    decoderOpt.zero_grad()

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
    decoderInput = torch.tensor(
        [[word2id["<sos>"]]] * batchSize, dtype=torch.long, device=device)

    decoderHidden = encoderHidden[-1]
    decoderOutput = None

    # Run through the decoder one step at a time. This seems to be common practice across
    # all of the seq2seq based implementations I've come across on GitHub.
    for t in range(ySeqlength):
        # compute the output of each decoder state
        DecoderOut = decoder(decoderInput, encoderOutputs,
                             decoderHidden, back=backwardOutput[:, t])

        # update variables
        decoderOutput, decoderHidden, pred_bow, infer_mu, infer_logvar, prior_mu, prior_logvar = DecoderOut

        # compute reference CBOW
        ref_bow = torch.FloatTensor(
            batchSize, vocabularySize).zero_().to(device)
        ref_bow.scatter_(1, y[:, t:], 1)

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
        decoderInput = y[:, t]
        decoderHidden = decoderHidden.squeeze(0)

    # calculate gradients
    loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(backwards.parameters(), gradientClip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradientClip)
    if useBOW:
        torch.nn.utils.clip_grad_norm_(cbow.parameters(), gradientClip)

    # gradient descent
    if useBOW:
        cbowOpt.step()
    decoderOpt.step()
    # priorOpt.step()
    # inferenceOpt.step()
    # attentionOpt.step()
    backwardsOpt.step()
    encoderOpt.step()

    return loss.item()/ySeqlength, ll_loss.item()/ySeqlength, kl_loss.item()/ySeqlength, aux_loss.item()/ySeqlength


def trainIteration(
        device,
        dataset,
        encoder,
        # attention,
        backwards,
        # inference,
        # prior,
        decoder,
        # cbow,
        iterations,
        word2id,
        criterion_r,
        criterion_bow,
        learningRate,
        gradientClip,
        useBOW,
        printEvery=10,
        plotEvery=100):

    plotLosses = []
    printLossTotal = 0
    plotLossTotal = 0

    encoderOpt   = optim.Adam(encoder.parameters(),   lr=learningRate)
    backwardsOpt = optim.Adam(backwards.parameters(), lr=learningRate)
    decoderOpt   = optim.Adam(decoder.parameters(),   lr=learningRate)

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
                # attention,
                backwards,
                # inference,
                # prior,
                decoder,
                encoderOpt,
                backwardsOpt,
                decoderOpt,
                word2id,
                criterion_r,
                criterion_bow,
                useBOW,
                gradientClip
            )

            print("Batch:", n, "Loss:", round(loss, 4), "LL:", round(
                ll, 4), "KL:", round(kl, 4), "AUX:", round(aux, 4))

            # increment our print and plot.
            printLossTotal += loss
            plotLossTotal += loss

            losses.append(loss)
            ll_losses.append(ll)
            kl_losses.append(kl)
            aux_losses.append(aux)

            # if batch_num % 10 == 0:
                # plotBatchLoss(j, ll_losses, kl_losses, aux_losses)

        saveModels(encoder, backwards, attention,
                   inference, prior, decoder, cbow)


if __name__ == "__main__":
    print("Loading parameters..", end=" ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        parameters = {
            'hiddenSize'			: 64,
            'latentSize'			: 32,
            'batchSize'				: 32,
            'iterations'			: 5,
            'learningRate'			: 0.0001,
            'gradientClip'			: 5,
            'useBOW'				: True,
            'bidirectionalEncoder'	: True,
            'reduction'             : 512
        }
    else:
        parameters = {
            'hiddenSize'			: 512,
            'latentSize'			: 400,
            'batchSize'				: 64,
            'iterations'			: 5,
            'learningRate'			: 0.0001,
            'gradientClip'			: 5,
            'useBOW'				: False,
            'bidirectionalEncoder'	: True,
            'reduction'             : 1
        }

    hiddenSize = parameters['hiddenSize']
    latentSize = parameters['latentSize']
    batchSize = parameters['batchSize']
    iterations = parameters['iterations']
    learningRate = parameters['learningRate']
    gradientClip = parameters['gradientClip']
    useBOW = parameters['useBOW']
    bidirectionalEncoder = parameters['bidirectionalEncoder']
    reduction = parameters['reduction']

    print(parameters)
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
    print("Batching Data..", end=" ")

    # shuffle data rows
    random.shuffle(train)
    random.shuffle(validation)

    trainx = [x[0] for x in train[::reduction]]
    trainy = [x[1] for x in train[::reduction]]
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

    embedding = nn.Embedding(
            num_embeddings=vocabularySize,
            embedding_dim=embeddingDim,
            padding_idx=paddingID,
        _weight=weightMatrix
        ).to(device)

    # we're using pretrained labels
    embedding.weight.requires_grad = False   

    modelEncoder = Encoder(embedding, vocabularySize,
                           paddingID, hiddenSize, bidirectionalEncoder).to(device)

    modelBackwards = Backwards(embedding, vocabularySize,
                               paddingID, hiddenSize, bidirectionalEncoder).to(device)

    modelDecoder = Decoder(embedding, vocabularySize,
                           paddingID, batchSize, maxReviewLength, hiddenSize, latentSize, bidirectionalEncoder).to(device)
    # modelBOW = CBOW(vocabularySize, latentSize).to(device)

    criterion_r = nn.NLLLoss(ignore_index=paddingID)
    criterion_bow = nn.BCEWithLogitsLoss()

    print("Done.")

    trainIteration(device,
                   traindata,
                   modelEncoder,
                #    modelAttention,
                   modelBackwards,
                #    modelInference,
                #    modelPrior,
                   modelDecoder,
                #    modelBOW,
                   iterations,
                   word2id,
                   criterion_r,
                   criterion_bow,
                   learningRate,
                   gradientClip,
                   useBOW,
                   printEvery=1000)

    saveModels(modelEncoder, modelBackwards, modelAttention,
               modelInference, modelPrior, modelDecoder, modelBOW)

    # GOAL
    # Batch: 0 Loss: 26731.2604 LL: 22.8447 KL: 26707.8917 AUX: 0.5253
