import torch
from torch import optim
import logging
import os
import json
import numpy as np
import random
from dataset import setup_dataloaders
from utils import setup_logger, parse_json, save_img_as_npz
import argparse
import sys
from torchvision.utils import save_image


import pandas as pd


import torch
import transformers
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from torch import cuda
from torch.nn import Identity
from torch.utils.data import Dataset, DataLoader


def setup_args():

    options = argparse.ArgumentParser()

    options.add_argument('--datadir', action="store", default="/projects/imagesets3/Cell_Painting_dataset/subset_bray/images/")
    options.add_argument('--train-metafile', action="store", default="data/metadata/datasplit_gen_train.csv")   #/projects/synsight/ethan/graph2pheno/graph2pheno/
    options.add_argument('--val-metafile', action="store", default="data/metadata/datasplit_gen_test_easy.csv")
    options.add_argument('--dataset', action="store", default="cell-painting")
    
    options.add_argument('--featfile', action="store", default=None)
    options.add_argument('--img-size', action="store", default=512, type=int)

    options.add_argument('--n_sample', default=30, type=int, help='number of samples')
    options.add_argument('--seed', action="store", default=42, type=int)
    options.add_argument('--batch-size', action="store", dest="batch_size", default=16, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=0, type=int)

    # gpu options
    options.add_argument('--use-gpu', action="store_false", default=True)

    options.add_argument('--save_dir',action="store",default='test_save')

    # debugging mode
    options.add_argument('--debug-mode', action="store_true", default=False)

    options.add_argument('--use_nce_loss', action="store",default=False )

    return options.parse_args()


# to work or visualize only with 3 channels: the ones taken represents the actin, WGA/phalloidin and nuclei (in RGB order)
def get_3c_image(image):
  return torch.stack((image[:,3,:,:],image[:,1,:,:],image[:,4,:,:])).permute(1,0,2,3)


args = setup_args()



  
trainloader , testloader = setup_dataloaders(args)


for batch_idx, (real_sample, cond) in enumerate(trainloader): # real sample are the real images and cond are the molecules (for of list len(bs))
    print(batch_idx)
    print(real_sample.shape)
    print(len(cond))

    

    break


      

