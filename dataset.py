import torchtext
from torchtext import data
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import Dataset
from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence

class Amazon(Dataset):
    def __init__(self,root="./amazon",split="train",N=1_000_000,MAX_VOCAB_SIZE = 50_000):
        if split=="train":
            self.df = pd.read_csv(f"{root}/{split}.csv",header=None).sample(N,random_state=42)
        else:
            self.df = pd.read_csv(f"{root}/{split}.csv",header=None)
        self.TEXT = data.Field(tokenize = 'spacy',
                              tokenizer_language = 'en_core_web_sm',
                              include_lengths = True,
                              pad_first=True)
        self.en = data.get_tokenizer("spacy",language='en_core_web_sm')
        # self.df["tokenized"] = self.df[2].apply(en)
        with Pool(2) as pool:
            self.df["tokenized"] = pool.map(self.en,self.df[2].values)
        self.TEXT.build_vocab(self.df["tokenized"], max_size = MAX_VOCAB_SIZE)
        
        self.X = self.df["tokenized"].values
        self.Y = self.df[0].values
        del self.df
    def __len__(self):
        return len(self.X)
    @staticmethod    
    def process(text_tok,TEXT):
        return torch.as_tensor([TEXT.vocab.stoi[x] if x in TEXT.vocab.stoi else 0 for x in text_tok ])
    def __getitem__(self,idx):
        x = self.process(self.X[idx],self.TEXT)
        y = self.Y[idx]
        return x,y
    
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for x,y in batch:
        src_batch.append(torch.as_tensor(x,dtype=torch.int64).view(-1,1))
        tgt_batch.append(torch.as_tensor(y-1,dtype=torch.int64).view(-1,1))
    src_batch = pad_sequence(src_batch, padding_value=1)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1)
    return src_batch.permute(1,0,2).squeeze(2), tgt_batch.squeeze()