from Bio.Seq import Seq
from Bio import SeqIO
from collections import Counter
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score,classification_report,confusion_matrix
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import json
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import nglview as nv
import os
import sklearn

import transformers
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch_optimizer as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from model_def import ProteinClassifier
from data_prep import ProteinSequenceDataset

max_len = 100 # providing the maximum length of a sequence
PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case = False)

def _get_train_data_loader(batch_size,training_dir):
    dataset = pd.read_csv(os.path.join(training_dir,'train.csv'))
    train_data = ProteinSequenceDataset(sequence=dataset.sequence.to_numpy(),
                                        targets=dataset.gene.to_numpy(),
                                        tokenizer=tokenizer,
                                        max_len=MAX_LEN)
    
    train_sampler = torch.utils.data.DistributedSampler(dataset,
                                                       num_replicas=dist.get_world_size(),
                                                       rank=dist.get_rank())
    train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True,
                                 sampler=train_sampler)
    return train_dataloader

def _get_test_data_loader(batch_size,training_dir):
    dataset = pd.read_csv(os.path.join(training_dir,'test.csv'))
    test_data = ProteinSequenceDataset(sequence=dataset.sequence.to_numpy(),
                                        targets=dataset.ggene.to_numpy(),
                                        tokenizer=tokenizer,
                                        max_len=MAX_LEN)
    
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=batch_size)
    return test_dataloader
                                       
def freeze(model,frozen_layers):
    modules = [model.bert.encoder.layer[:frozen_layers]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad=False
                                       
def train(args):
    use_cuda=args.num_gpus > 0
    device= torch.device('cuda' if use_cuda else 'cpu')
    world_size= dist.get_world_size()
    rank = dist.get_rank()
    local_rank=dist.get_local_rank()
    
    #set the random seed for generating random numbers
    torch.manual_seed(args.seed)
    is use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    train_loader = _get_train_data_loader(args.batch_size,args.data_dir)
    
    if rank==0:
        test_loader = _get_test_data_loader(args.test_batch_size,args.test)
        print('max length of sequence: ',MAX_LEN)
        print('freezing {} layers'.format(args.frozen_layers))
        print('Model used: ',PRE_TRAINED_MODEL_NAME)
        
    logger.debug(
        'Prcosses {}/{} ({.0f}%) of train data'.format(len(train_loader.sampler),
                                                      len(train_loader.dataset),
                                                      100.0 = len(train_loader.sampler)/len(train_loader.dataset),
                                                      )
    )
    model = ProteinClassifier(args.num_labels) # the number of output labels
    freeze(model,args.frozen_layers)
    model = DDP(model.to(device),broadcast_buffers=False)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    
    optimizer = optim.Lamb(model.parameters(),
                          lr = args.lr * dist.get_world_size(),
                          betas = (0.9,0.999),
                          eps=args.epsilon,
                          weight_decay=args.weight_dacay)
    total_steps=len(train_loader.dataset)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=total_steps)
    
    loss_fn == nn.CrossEntropyLoss().to(device)
    
    for epoch in enage(1,args.epochs+1):
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)
            
            outputs = model(b_input_ids.attention_mask=b_input_mask)
            loss = loss_fn(outputs,b_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
            #modified based on their gradients, the learning rate etc
            optimizer.step()
            optimizer.zero_grad()
            
            if step% args.log_interval == 0 and rank == 0:
                logger.info(
                'collections data from master Node: \n Train Epoch: {} [{}/{} ({:.0f}%)] Training Loss: {:.6f}'.format(
                    epoch,
                step * len(batch['input_ids'])%world_size,
                len(train_loader.dataset),
                100.0 *step/len(train_loader),
                loss.item(),
                ))
            
            if args.verbose:
                print('Batch',step,'from rank',rank)
        if rank == 0:
            test(model,test_loader,device)
        scheduler.step()
    if rank == 0:
        model_save = model.module if hasattr(model,'module') else model
        save_model(model_save,args.model_dir)

def save_model(model,model_dir):
    path = os.path.join(model_dir,'model.pth')
    
    torch.save(model.state_dict(),path)
    logger.info(f'Saving Model: {path} \n')

def test(model,test_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    loss_fn = nn.CrossEntropyLoss().to(device)
    tmp_eval_accuracy, eval_accuracy = 0,0
    
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            