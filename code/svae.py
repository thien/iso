import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from vad_utils import kl_anneal_function

"""
Adapted from https://github.com/timbmg/Sentence-VAE
"""

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def KL(mean, logv, anneal_function, step, k, x0):
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    kl_weight = kl_anneal_function(anneal_function, step, k, x0)
    return kl_loss, kl_weight

def recon_loss(logp, target, length, NLL):
    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))
    # Negative Log Likelihood
    return NLL(logp, target)

class SentenceVAE(nn.Module):

    def __init__(self, embedding, vocab_size, embedding_size, hidden_size, latent_size, word_dropout_rate,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = embedding

        self.word_dropout_rate = word_dropout_rate

        self.encoder_rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.decoder_rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)

        # self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * 2 * num_layers, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * 2 * num_layers, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * num_layers)
        self.outputs2vocab = nn.Linear(hidden_size, vocab_size)

        self.nll = torch.nn.NLLLoss(reduction='sum', ignore_index=self.pad_idx)

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # print(packed_input)

        # empty = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        _, hidden = self.encoder_rnn(packed_input)

        # if self.bidirectional or self.num_layers > 1:
        #     # flatten hidden state
        hidden = hidden.view(batch_size, self.hidden_size*self.num_layers*2)
        # else:
        #     hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        # if self.bidirectional or self.num_layers > 1:
        #     # unflatten hidden state
        #     hidden = hidden.view(num_layers, batch_size, self.hidden_size)
        # else:
        hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        # input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z
