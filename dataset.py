import os
import h5py
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from bpe import create_bpe_tokenizer


__TOTAL_IMAGE_TOKENS__ = 64

__SOS_IMAGE_TOKEN__  = 64
__EOS_IMAGE_TOKEN__  = 65
__MASK_IMAGE_TOKEN__ = 66

__SOS_TEXT_TOKEN__   = 67
__EOS_TEXT_TOKEN__   = 68
__PAD_TEXT_TOKEN__   = 69


class Language:
    
    def __init__(self, name):
        
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            __SOS_IMAGE_TOKEN__ : '[SOS_image]',
            __EOS_IMAGE_TOKEN__ : '[EOS_image]', 
            __MASK_IMAGE_TOKEN__: '[MASK]',
            __SOS_TEXT_TOKEN__  : '[SOS_text]', 
            __EOS_TEXT_TOKEN__  : '[EOS_text]',
            __PAD_TEXT_TOKEN__  : '[PAD]'
        }

        self.n_words = __TOTAL_IMAGE_TOKENS__ + 6 

    def addSentence(self, sentence):
        for word in sentence: self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, max_text_length):
    
    indexes = np.ones(max_text_length, dtype=np.int32) * __PAD_TEXT_TOKEN__
    
    indexes[0] = __SOS_TEXT_TOKEN__
    ids        = indexesFromSentence(lang, sentence)
    nonpad_len = min([max_text_length - 1, len(ids)])
    
    for i in range(nonpad_len):
        indexes[i+1] = ids[i]

    indexes[nonpad_len] = __EOS_TEXT_TOKEN__

    return torch.tensor(indexes, dtype=torch.long)



class Text2ImageDataset(Dataset):

    def __init__(
        self, datasetFile,
        annotationsFile=None,
        max_text_length=128,
        split=0,
    ):
        """
        :args split: flag, indicating the part of the dataset
        :type split: int

        :args datasetFile: filename of the hdf5 file with flowers dataset
        :type datasetFile: string

        :args annotationsFile: 
            filename of dataset annotations file in subword_nmt-compatible format
        :type annotationsFile: string

        """
        self.datasetFile     = datasetFile
        self.annotationsFile = annotationsFile
        self.max_text_length = max_text_length
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = lambda x: int(np.array(x))

        self.dataset = h5py.File(self.datasetFile, mode='r')
        self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]
        
        self.bpe_tokenizer        = self.__bind_bpe_tokenizer()
        self.annotations_language = self.__create_annotations_language()


        self.image_tokens = 67
        self.text_tokens  = self.annotations_language.n_words - self.image_tokens


    def __bind_bpe_tokenizer(self):

        if self.annotationsFile is None:
            with open('annotations.txt', 'w') as f: 
                for sample_name in self.dataset_keys: 
                    f.writelines(str(np.array(self.dataset[self.split][sample_name]['txt']).astype(str)))

            self.annotationsFile = 'annotations.txt'

        return create_bpe_tokenizer(self.annotationsFile)

    def __create_annotations_language(self):

        annotations_language = Language('annotations')
        with open(self.annotationsFile, 'r') as f: annotations = f.readlines()

        for s in annotations: 
            annotations_language.addSentence(self.bpe_tokenizer.process_line(s).split(' '))

        return annotations_language

    def __len__(self): return len(self.dataset_keys)

    def __getitem__(self, idx):

        example_name = self.dataset_keys[idx]
        example      = self.dataset[self.split][example_name]
        
        vqvae_code = [__SOS_IMAGE_TOKEN__] + list(np.array(example['vqvae_code']).flatten()) + [__EOS_IMAGE_TOKEN__]
        vqvae_code = torch.tensor(vqvae_code)
        
        txt = str(np.array(example['txt']).astype(str))
        txt = tensorFromSentence(
            self.annotations_language, 
            self.bpe_tokenizer.process_line(txt), 
            self.max_text_length
        )

        sample = torch.cat([txt, vqvae_code], dim=0)

        return sample  