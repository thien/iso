import os
import json
import time
import gzip
import bcolz
import numpy as np
import re
import copy
from multiprocessing import Pool
from tqdm import tqdm

try:
    import cPickle as pickle
except:
    import pickle


class DataProcessor:
    def __init__(self, picklelocation, verbose=False):
        self.rawFilepath = picklelocation
        self.dataset = None
        self.verbose = verbose
        self.glove = None
    
    def loadDataset(self):
        start = time.clock()
        self.dataset = pickle.load(open(self.rawFilepath, "rb"))
        if self.verbose:
            duration = time.clock() - start
            print("Raw dataset loaded in", duration, "seconds.")

    def loadGlove(self, glove_path, dim=50):
        acceptedDimensions = [50, 100, 200, 300]
        if dim not in acceptedDimensions:
            print("You didn't choose a right dimension.")
            print("Try one of these:", acceptedDimensions)
            return None
        pickleWordFile = f'{glove_path}/6B.'+str(dim)+'_words.pkl'
        pickleIdFile   = f'{glove_path}/6B.'+str(dim)+'_idx.pkl'
        pickleDatFile  = f'{glove_path}/glove.6B.'+str(dim)+'.dat'
        pickleDataset  = f'{glove_path}/glove.6B.'+str(dim)+'d.txt'
        
        if os.path.isfile(pickleWordFile):
            # check if we've made the outputs before
            if self.verbose:
                print("Preloading files..", end=" ")
            vectors = bcolz.open(pickleDatFile)[:]
            words = pickle.load(open(pickleWordFile, 'rb'))
            word2idx = pickle.load(open(pickleIdFile, 'rb'))
            glove = {w: vectors[word2idx[w]] for w in words}
            if self.verbose:
                print("Done.")
            self.glove = glove
        else:
            print("Doesn't work.", end=" ")

    @staticmethod
    def preprocess(paragraph):
        # split paragraph by full stops
        paragraph = paragraph.lower()
        paragraph = re.sub("([,!?()-+&£$.%*'])", r' \1 ', paragraph)
        paragraph = re.sub('\s{2,}', ' ', paragraph)
        paragraph = paragraph.split(" ")
        # remove empty string
        return paragraph

    @staticmethod
    def padSentence(words, maxLength, eosString="<eos>", padString="<pad>"):
        if len(words) > maxLength:
            return words[:maxLength-1] + [eosString]
        else:
            return words + [padString for i in range(maxLength-1 - len(words))]
    
    @staticmethod
    def discretise(value, word):
        return word + "_" + str(value)

    def handle(self, itemID,
            minWords=6, 
            minEntries=5, 
            maxSummaryLength=10, 
            maxReviewLength=60):
        """
        Filters words out based on whether they're in the GloVe dataset or not.
        
        Parameters:
        
        """

        reviews = []

        dataset = self.dataset
        wordbase = self.glove.keys()
        printDebug = False

        # check if there are more than 5 reviews.
        if len(dataset[itemID]) > minEntries:
            review = []
            for i in range(len(dataset[itemID])):
                # initialise variables
                entry = dataset[itemID][i]

                # preprocess review
                words = self.preprocess(entry['reviewText'])
                words = [w for w in words if w in wordbase]
                words = words[:maxReviewLength]
                # preprocess summary
                summary = self.preprocess(entry['summary'])
                summary = [w for w in summary if w in wordbase]
                summary = summary[:maxSummaryLength]

                # visualise
                if printDebug:
                    print(dataset[itemID][i])
                    print(numWords, words, summary, "\n")

                # if theres more than 6 tokens, keep the first 50 tokens.
                if len(words) > minWords:
                    # add padding tokens here
                    summary  = self.padSentence(summary, maxSummaryLength)
                    words    = self.padSentence(words,   maxReviewLength)
                    # also need to process rating, polarity and item_id
                    rating   = [self.discretise(entry['overall'], "rating")]
                    # process polarity
                    polarity = np.round(np.tanh(entry['helpful'][0]-entry['helpful'][1]),1)
                    polarity = [self.discretise(polarity, "polarity")]
                    reviewID = [itemID]
                    entry = reviewID + summary + rating + polarity + words
                    review.append(entry)

            # check if theres less than 5 filtered reviews
            if len(review) > minEntries:
                reviews.append(review)    
        return reviews

    @staticmethod
    def imap_unordered_bar(func, args, n_processes = 7):
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

if __name__ == "__main__":
    pickleFile = '../Datasets/Reviews/dataset.pkl'
    glovePath = "/media/data/Datasets/glove"
    gloveDimension = 50

    processor = DataProcessor(pickleFile, verbose=True)
    processor.loadDataset()
    processor.loadGlove(glovePath, dim=gloveDimension)
    processor.imap_unordered_bar(processor.handle,list(processor.dataset.keys()))