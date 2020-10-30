import deepspeed

from routing_transformer import RoutingTransformerLM
from routing_transformer.autoregressive_wrapper import AutoregressiveWrapper

import argparse
import random
import tqdm
import gzip
import numpy as np
import torch
from torch import load
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import lm_dataformat as lmd
from transformers import GPT2Tokenizer
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
import shutil
import time
#from utils import save_checkpoint
#from utils import load_checkpoint

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')
    parser.add_argument('--chckpt', type=int, default=-1, help="When set to some multiple of \"GENERATE_EVERY\", signifies which batch\
            the model should be loaded from. (default: -1)")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

# constants

VALIDATE_EVERY  = 500
VALIDATE_SIZE = 50
GENERATE_EVERY  = 10000
SAVE_EVERY = 2500
GENERATE_LENGTH = 256
SEQ_LEN = 2048

# helpers

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model

model = RoutingTransformerLM(
    num_tokens = tokenizer.vocab_size,
    dim = 768,
    depth = 12,
    max_seq_len = SEQ_LEN,
    heads = 12,
    causal = True,
    window_size = 256,
    local_attn_window_size = 512,
    reversible = False,
    ff_chunks = 2,
    ff_glu = True,
    attn_dropout = 0.0,
    rel_pos_emb = False,
    n_local_attn_heads = (12,12,12,12,12,12,12,12,12,12,12,12),
    _register_kmeans_update = False,
    tie_embedding = True,
    use_absolute_pos_emb = True,
    kmeans_ema_decay = 0.9999,
    commitment_factor = 1e-5,
)

model = AutoregressiveWrapper(model)
model.cuda()

class TextSamplerDataset(Dataset):
    def __init__(self, directory, cache_size = 100):
        super().__init__()
        self.c = cache_size
        self.indx = 0
        #Some magic. Dynamic shuffling and caching
        self.cache_dict =   [True] * self.c
        self.cache = [None] * self.c
        self.docs = lmd.Reader(directory).stream_data()
        self.val_list = list()
        self.state = 0
        #Preload the first k elements
        for i in range(self.c):
            self.cache[i] = self.read()

    #Read the next element from stream
    def read(self):
        text = next(self.docs)
        tok = tokenizer(text, return_tensors='pt')['input_ids'].long()
        mask = torch.ones_like(tok).bool()

        if tok.size(1) >= SEQ_LEN:
            tok = tok[0:,0:SEQ_LEN]
            mask = mask[0:,0:SEQ_LEN]
        else:
            tok = F.pad(tok, pad=(0, SEQ_LEN - tok.size(1)), mode='constant', value=0)
            mask = F.pad(mask, pad=(0, SEQ_LEN - mask.size(1)), mode='constant', value=0)

        if len(self.val_list) <= VALIDATE_SIZE:
            self.val_list.append((tok.squeeze(), mask.squeeze()))
            return self.read()

        self.state += 1
        return tok.squeeze(), mask.squeeze()
    def __getitem__(self, index): 
        true_i = index % self.c
        #If our current element is missing, look at all prior ones and fill to this point.
        if not self.cache_dict[true_i]:
            for i,b in enumerate(self.cache_dict[0:true_i + 1]):
                if not b:
                    self.cache[i] = self.read()
                    self.cache_dict[i] = True

        return self.cache[true_i]
    def fill_val(self):
        self.read()
    def __len__(self):
        return int(202952221 * 0.98)
    def get_val(self):
        return self.val_list
    def clear_val(self):
        self.val_list  = list()
    def resume(self, new_state):
        for _ in range(new_state): next(self.docs)
dataset = TextSamplerDataset("/media/ambient/LargeDatasetsSSD/pile/The-Pile/pile_output/")

# setup deepspeed

cmd_args = add_argument()
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(),  training_data=dataset)

s = 0
if cmd_args.chckpt is not -1:
    model_engine.load_checkpoint("models/", str(cmd_args.chckpt))
    dataset.resume(cmd_args.chckpt + 1)
    s = cmd_args.chckpt + 1

for i, (data, mask) in tqdm(enumerate(trainloader, start=s), initial=s):
    model_engine.train()

    data = data.to(model_engine.local_rank)
    loss = model_engine(data, return_loss = True, randomly_truncate_sequence = True)
    model_engine.backward(loss)
    check = model_engine.is_gradient_accumulation_boundary()
    model_engine.step()
    if check:
        model.base_net.update_kmeans()
    if (i+1) % VALIDATE_EVERY == 0:
        val_dataset = dataset.get_val()
        torch.distributed.barrier()
        dataset.clear_val()
        dataset.read()

    if (i+1) % VALIDATE_EVERY == 0 and model_engine.local_rank==0:
        model.eval()
        val_indx = 0
        print(loss)
        with torch.no_grad():

            loss_sample = 0
            for inp,_ in val_dataset:
                loss_sample += model(inp[None, :].cuda(), return_loss = True)
            loss_s = loss_sample.item() / float(len(val_dataset))
            print(f'validation loss: {loss_s}')
    
    if i != 0 and ((i % SAVE_EVERY) == 0):
        torch.distributed.barrier()
        #print(i)
        # path = "./models/"+str(i) + "/"

        try:
        	os.mkdir("models")
        except:
        	pass
        model_engine.save_checkpoint("models/", str(i))


    if i != 0 and ((i % GENERATE_EVERY) == 0) and model_engine.local_rank == 0:
        #model_engine.save_checkpoint("./", "deepspeed")

        #pickle.dump(optimizer, open(path+"optimizer.pkl", "wb"))

        #print('donesave')
        #import sys
        #sys.exit()
        model.eval()
        val_dataset = dataset.get_val()
        with torch.no_grad():
            inp, _ = random.choice(val_dataset)
            #print('huh')
            prime = tokenizer.decode(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp.cuda(), GENERATE_LENGTH)
            output_str = tokenizer.decode(sample)
            print(output_str)
    torch.distributed.barrier()
