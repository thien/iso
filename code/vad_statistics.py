# plotting
import matplotlib
# we're only writing to file.
matplotlib.use('Agg')
import argparse
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import csv
import re
import json
import pickle
import torch
from vad_utils import loadDataset, batchData, convertRealID2Word, prepDataset
from tqdm import tqdm
from multiprocessing import Pool

from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from nltk.translate.bleu_score import sentence_bleu
plt.rc('text',usetex=True)
font = {'family':'serif','size':16}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':14})
plt.rc('figure', **{'dpi': 200}) 


class Statistics:
    def __init__(self, model_folder, verbose=False):
        # configuration variables
        self.model_folder = model_folder
        self.parent_foldername = "models"
        self.model_results_folder = "results"
        self.model_outputs_folder = "outputs"
        self.results_csv_re = r"([a-z]+_)([0-9]+)([a-z]+.csv)"
        self.output_re = r"([a-z]+_)([0-9]+)(.csv)"
        # location to save charts
        self.charts_foldername = "results_charts"
        self.stat_chart_filename = "bleu_rouge"
        self.charts_ext = ".pdf"
        self.model_parameters_filename = "model_parameters.json"
        self.dataset_parameters_filename = "dataset_parameters.json"
        
        # loaded when initiated
        self.verb = verbose
        self.parameters = None
        self.useKL = True
        self.useBOW = True
        self.results = None
        self.dataset = None

        self.valx_tokens = None
        self.valy_tokens = None
        self.outputs = None
        self.id2word = None

        self.statistics = None

    # load the csv containing the results
    def loadResults(self):
        """
        Loads csvs of all epochs from a given model_folder name.
        Returns a concatenation of results.
        """
        # setup filepath
        folder = os.path.join(self.parent_foldername, self.model_folder, self.model_results_folder)
        files = os.listdir(folder)
        
        # retrieve .csv files.
        if len(files) < 1: return
     
        # retrieve only csvs
        files = [i for i in files if (i[-4:].lower() == ".csv")]
        if self.verb:
            print("Loading", len(files), "results..", end=" ")
        parts = re.match(self.results_csv_re, files[-1], re.I).groups()
        
        # sort by epochs (increasing order)
        numbs = sorted([int(re.findall('\d+', i )[0]) for i in files])
        
        results = None
        for number in numbs:
            # create filename
            file = parts[0] + str(number) + parts[2]
            filepath = os.path.join(folder, file)
            
            ents = np.loadtxt(filepath, delimiter=", ")
            if results is None:
                results = ents
            else:
                results = np.vstack((results,ents))
        self.results = results
        self.printFinish()

    def plotChart(self, step=100, ylim=6):
        if self.verb:
            print("Plotting loss chart..", end=" ")
        if self.results is None:
            raise Exception('There are no results to plot from. Please run loadResults()')
        recon = self.results[:,0]
        x = np.arange((recon.shape[0]))
        best_fit = np.poly1d(np.polyfit(x, recon, 4))

        plt.figure(figsize=(8,5))

        plt.plot(self.results[::step, 0], label="Reconstruction Loss")
        if self.useKL:
            plt.plot(self.results[::step, 1], label="KL Loss")
        if self.useBOW:
            plt.plot(self.results[::step, 2], label="BOW Loss")

        plt.plot(best_fit(x)[::step], label="Reconstruction Loss Best Fit")
        plt.ylim(0,ylim)
        plt.legend()
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        # initiate chart save location
        filename = "model_loss" + self.charts_ext
        directory = os.path.join(self.parent_foldername, self.model_folder, self.charts_foldername)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
  
        self.printFinish()

    def loadModelParameters(self):
        if self.verb:
            print("Loading model parameters..", end=" ")
        # we want to load the model parameters
        filepath = os.path.join(self.parent_foldername,self.model_folder, self.model_parameters_filename)
        with open(filepath, "r") as f:
            self.parameters = json.load(f)
        
        self.printFinish()

    def loadPenn(self):
        if self.verb:
            print("Loading Penn data..", end=" ")
        # shallow integration of the sentenceVAE codebase s.t. we can
        # load the penn dataset.
        from penn_utils import to_var, idx2word, expierment_name, PTB
        # from sendata_utils import to_var, idx2word, expierment_name, PTB
        from collections import OrderedDict, defaultdict

        splits = ['valid']

        valdata = PTB(
                data_dir='../Datasets/Penn',
                split='valid',
                create_data='store_true',
                max_sequence_length=60,
                min_occ=1
            )

        with open('../Datasets/Penn/ptb.vocab.json', 'r') as file:
            vocab = json.load(file)

        word2id, id2word = vocab['w2i'], vocab['i2w']
        self.id2word = {int(key) : id2word[key] for key in id2word.keys()}
        
        self.val_loader = DataLoader(
            dataset=valdata,
            batch_size=self.parameters['batch_size'],
            shuffle=False,
            num_workers=cpu_count()-1,
            pin_memory=torch.cuda.is_available()
        )

        self.parseDataLoader()

        self.printFinish()

    # now we're looking at loading the dataset from each model.
    def loadDatasetFromModel(self):
        if self.verb:
            print("Loading dataset from model configurations..", end=" ")
     
        if "dataset" in self.parameters:
            if self.parameters['dataset'].lower() == "penn":
                self.loadPenn()
                return

        batchSize = self.parameters['batch_size']
        reduction = self.parameters['reduction']

        # load dataset parameters so we know the location of the datasets.
        filepath = os.path.join(self.parent_foldername,self.model_folder, self.dataset_parameters_filename)
        
        with open(filepath, "r") as f:
            dataset_parameters = json.load(f)

        # load dataset
        dataset_filepath = dataset_parameters['datasetFile']

        dataset = loadDataset(path=dataset_filepath)

        word2id, id2word = dataset['word2id'], dataset['id2word']
        self.id2word = id2word
        self.word2id = word2id
        paddingID, sosID = word2id['<pad>'], word2id['<sos>']
        cutoff = dataset['cutoff']

        # load validation data.
        val = dataset['validation']

        self.printFinish()

        valdata = prepDataset(val, paddingID, cutoff, train=False, step=reduction)

        self.val_loader = DataLoader(
            dataset=valdata,
            batch_size=batchSize,
            shuffle=False,
            num_workers=cpu_count()-1
        )

        self.parseDataLoader()

        if self.verb:
            print("Loaded",len(valdata),"input validation sequences.")

    def parseDataLoader(self):
        self.valx_tokens = [convertRealID2Word(self.id2word, batch['input'], toString=False) for batch in self.val_loader]
        self.valy_tokens = [convertRealID2Word(self.id2word, batch['target'], toString=False) for batch in self.val_loader]

    # now we need to load the model results
    def loadModelOutputs(self):
        """
        returns a list of batched responses.
        outputs =   [ (epoch)
                        [ (batch)
                            [ (response) ]
                        ]
                    ]
        """

        if self.verb:
            print("Loading model responses..", end=" ")
    
        folder = os.path.join(self.parent_foldername, self.model_folder, self.model_outputs_folder)
        files = os.listdir(folder)
        
        # retrieve .csv files.
        if len(files) < 1: return
        files = [i for i in files if (i[-4:].lower() == ".csv")]
        parts = re.match(self.output_re, files[0], re.I).groups()
        # sort by epochs (increasing order)
        numbs = sorted([int(re.findall('\d+', i )[0]) for i in files])
        results = None
        
        outputs = []
        for number in numbs:
            # create filename
            file = parts[0] + str(number) + parts[2]
            filepath = os.path.join(folder, file)
            
            batches = []
            with open(filepath) as csvDataFile:
                csvReader = csv.reader(csvDataFile, delimiter="\t")
                for batch in csvReader:
                    batches.append(batch)
            outputs.append(batches)
        self.outputs = outputs
        self.printFinish()

    def printFinish(self):
        if self.verb:
            print("Done.")

    # Rouge Calculation
    @staticmethod
    def calcRouge(act, pred, n=1):
        # create n-grams
        ngram_ref  = [tuple(act[i:i+n]) for  i in range(len(act)-(n-1))]
        ngram_pred = [tuple(pred[i:i+n]) for i in range(len(pred)-(n-1))]
        # create dictionary of reference n-grams to count against.
        ref = {}
        for gram in ngram_ref:
            if gram not in ref:
                ref[gram] = 0
            ref[gram] = 1
        # compute rouge (recall)
        count = sum([1 if gram in ref else 0 for gram in set(ngram_pred)])
        if count > 0:
            return count / len(ngram_ref)
        return count

    # Mean averaged rouge with 1-3 grams
    def mean_rouge(self, act, pred, n_min=1, n_max=3):
        rouges = [self.calcRouge(act,pred,n) for n in range(n_min, n_max+1)]
        return sum(rouges)/len(rouges)

    # calculate bleu and rouge scores.
    def processPredictions(self, epoch=-1):
        bleus1 = []
        bleus2 = []
        rouges_1 = []
        rouges_2 = []
        rouges_3 = []

        for batch_num in range(len(self.valy_tokens)-1):
            for seq_num in range(len(self.valy_tokens[0])):
                condition = self.valx_tokens[batch_num][seq_num]
                actual    = self.valy_tokens[batch_num][seq_num]

                try:
                    outputs   = self.outputs[epoch][batch_num][seq_num]
                    predicted = outputs.split(" ")
                except:
                    predicted = []
                bleu1 = sentence_bleu([actual], predicted, weights=[1])
                bleu2 = sentence_bleu([actual], predicted, weights=[0.5,0.5])
                rouge1 = self.calcRouge(actual, predicted, n=1)
                rouge2 = self.calcRouge(actual, predicted, n=2)
                rouge3 = self.calcRouge(actual, predicted, n=3)
                bleus1.append(bleu1)
                bleus2.append(bleu2)
                rouges_1.append(rouge1)
                rouges_2.append(rouge2)
                rouges_3.append(rouge3)

        stats = {
            'bleu1' : np.mean(bleus1),
            'bleu2' : np.mean(bleus2),
            'rouge_1': np.mean(rouges_1),
            'rouge_2': np.mean(rouges_2),
            'rouge_3': np.mean(rouges_3)
        }

        # bleu: precision; rouge: recall
        stats['f1'] = (2 * stats['bleu1'] * stats['rouge_1']) / (stats['bleu1'] + stats['rouge_1']) 

        return stats

    def chartBLEUROUGE(self, statistics):
        bloos = [i['bleu1'] for i in statistics]
        bloos2 = [i['bleu2'] for i in statistics]
        rouges_1 = [i['rouge_1'] for i in statistics]
        rouges_2 = [i['rouge_2'] for i in statistics]
        rouges_3 = [i['rouge_3'] for i in statistics]
        f1 = [i['f1'] for i in statistics]

        plt.figure(figsize=(8,5))
        plt.plot(bloos, label="BLEU 1")
        plt.plot(bloos2, label="BLEU 2")
        plt.plot(rouges_1, label="ROUGE 1")
        plt.plot(rouges_2, label="ROUGE 2")
        plt.plot(rouges_3, label="ROUGE 3")
        plt.plot(f1, label="F1")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Percentage")
        plt.title("Emperical Measurements over Epochs")
        
        # time to save this.
        filename = self.stat_chart_filename + self.charts_ext
        directory = os.path.join(self.parent_foldername, self.model_folder, self.charts_foldername)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        
    def dumpStats(self, stats):
        directory = os.path.join(self.parent_foldername, self.model_folder)
        filename = self.stat_chart_filename + ".json"
        filepath = os.path.join(directory, filename)
        # dump the json of stats
        with open(filepath,"w") as fw:
            json.dump(stats, fw)

    @staticmethod
    def processItems(func, args, n_processes = 7):
        p = Pool(n_processes)
        res_list = []
        with tqdm(total = len(args)) as pbar:
            for i, res in enumerate(p.imap_unordered(func, args)):
                pbar.update()
                res_list.append(res)
        pbar.close()
        p.close()
        p.join()
        return res_list

    def batchComputeBLEUROUGE(self):
        self.bleuRougeStats = self.processItems(self.processPredictions, [ep for ep in range(len(self.outputs))])
        return self.bleuRougeStats

    def express(self):
        """
        Performs all measurement operations from scratch

        """
        print("Processing statistics for", self.model_folder)
        self.loadModelParameters()
        self.loadResults()
        self.loadDatasetFromModel()
        self.loadModelOutputs()
 
        self.plotChart(step=50, ylim=5)
        if self.verb:
            print("Calculating BLEU/ROUGE Scores..", end=" ")
        print(len(self.outputs))
        stats = self.processItems(self.processPredictions, [ep for ep in range(len(self.outputs))])
        self.printFinish()
        self.chartBLEUROUGE(stats)
        self.dumpStats(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir','-m', type=str)

    args = parser.parse_args()
    args.model_dir 
    s = Statistics(args.model_dir)
    s.verb = True
    s.express()
