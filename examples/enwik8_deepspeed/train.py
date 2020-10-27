import deepspeed

from routing_transformer import RoutingTransformerLM
from routing_transformer.autoregressive_wrapper import AutoregressiveWrapper

import argparse
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import lm_dataformat as lmd
from transformers import GPT2Tokenizer
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
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

VALIDATE_EVERY  = 1000
GENERATE_EVERY  = 5000
GENERATE_LENGTH = 256
SEQ_LEN = 4096

# helpers

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model

model = RoutingTransformerLM(
    num_tokens = tokenizer.vocab_size,
    dim = 512,
    depth = 8,
    max_seq_len = SEQ_LEN,
    heads = 8,
    causal = True,
    window_size = 128,
    reversible = True,
    ff_chunks = 2,
    attn_dropout = 0.1,
    rel_pos_emb = False,
    n_local_attn_heads = (8,8,8,8,8,8,6,6)
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
        tok = tokenizer(text, return_tensors='pt')['input_ids']
        mask = torch.ones_like(tok).bool()

        if tok.size(1) >= SEQ_LEN:
            tok = tok[0:,0:SEQ_LEN]
            mask = mask[0:,0:SEQ_LEN]
        else:
            tok = F.pad(tok, pad=(0, SEQ_LEN - tok.size(1)), mode='constant', value=0)
            mask = F.pad(mask, pad=(0, SEQ_LEN - mask.size(1)), mode='constant', value=0)

        if len(self.val_list) <= 50:
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

# training
if cmd_args.chckpt is not -1:
    checkpoint = torch.load("models/"+ str(cmd_args.load) + "/"+"model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    dataset.resume(checkpoint['doc_pos'])
for i, (data, mask) in tqdm(enumerate(trainloader)):
    model_engine.train()

    data = data.to(model_engine.local_rank)
    loss = model_engine(data, return_loss = True, randomly_truncate_sequence = True)
    model_engine.backward(loss)
    model_engine.step()
    #print(loss.item())

    if (i+1) % VALIDATE_EVERY == 0 and model_engine.local_rank == 0:
        model.eval()
        val_dataset = dataset.get_val()
        val_indx = 0
        with torch.no_grad():
            val_indx, (inp, _) = random.choice(list(enumerate(val_dataset)))
            loss = model(inp[None, :].cuda(), return_loss = True)
            print(f'validation loss: {loss.item()}')
        dataset.val_list[val_indx] = dataset.read()

    torch.distributed.barrier()
    if i != 0 and model_engine.local_rank == 0 and i % GENERATE_EVERY == 0:
        path = "models/"+str(i) + "/"
        try:
            os.mkdir(path)
        except:
            #Dir already exists. Overwrite the file
            pass
        torch.save({
            'iteration': i,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'doc_pos' : dataset.state
        }, path+"model.pt")
        torch.distributed.barrier()


        model.eval()
        val_dataset = dataset.get_val()
        inp, _ = random.choice(val_dataset)
        prime = tokenizer.decode(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp.cuda(), GENERATE_LENGTH)
        output_str = tokenizer.decode(sample)
        print(output_str)


