import os
from io import open
import torch
import pickle
# plotting
# plotting
import matplotlib
matplotlib.use('Agg')

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def loadDataset(path='../Datasets/Reviews/dataset_ready.pkl'):
    return pickle.load(open(path, 'rb'))

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
    KL = torch.sum(KL)

    # KL Annealing
    kl_weight = (batch_num+1)/10000
    # kl_weight = 1
    # weighted_KL = KL * kl_weight
    weighted_KL = KL

    # compute auxillary loss
    aux = criterion_bow(pred_bow, ref_bow)
    # weight auxillary loss
    alpha = 0.5
    weighted_aux = aux * alpha

    return LL + weighted_KL + weighted_aux, LL, weighted_KL, weighted_aux


def plotBatchLoss(iteration, losses, kl, aux):
    x = [i for i in range(1, len(losses)+1)]

    labels = ["KL", "Auxiliary", "LL"]

    plt.stackplot(x, kl, aux, losses, labels=labels)
    plt.legend()
    plt.ylim(top=15)
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
