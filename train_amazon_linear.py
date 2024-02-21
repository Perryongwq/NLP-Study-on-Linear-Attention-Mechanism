from fast_transformers.attention import FullAttention,LinearAttention,CausalLinearAttention
import torch
from dataset import Amazon,collate_fn
from torch.utils.data import DataLoader
from classifier import Amazon_Classifier
from tensorboardX import SummaryWriter
from tqdm import tqdm

from pynvml import *


EPOCHS = 30
N_BATCH = 64

if __name__=="__main__":
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    torch.manual_seed(0)
    
    dataset=Amazon(split="train",N=1_000_000)
    dataset_val=Amazon(split="test")
    train_dataloader=DataLoader(dataset,batch_size=N_BATCH,collate_fn=collate_fn)
    val_dataloader=DataLoader(dataset_val,batch_size=N_BATCH,collate_fn=collate_fn)
    
    writer = SummaryWriter(logdir = f"./tensorboard_logs/LinearAttention")


    model = Amazon_Classifier(layer=LinearAttention,dim=256,n_layers=3,n_heads=8,dim_feedfwd=512,causal=False)
    model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9,weight_decay=1e-04)

    # model.train()
    
    itr=0
    for e in tqdm(range(EPOCHS)):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        model.train()
        for src, tgt in train_dataloader:
            src=src.cuda()
            tgt=tgt.cuda()
            
            logits = model(src)
            
            # print(logits.shape,tgt.shape)
            optimizer.zero_grad()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt)
            loss.backward()
            optimizer.step()
            
            itr+=1
            with torch.no_grad():
                equals = (logits.sigmoid().argmax(1)==tgt).reshape(-1,1).detach().cpu()
                training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                training_loss += src.shape[0] * loss.item()
                training_samples += src.shape[0]
            if itr%50:
                train_loss = (training_loss/training_samples)
                train_acc = (training_acc/training_samples)
                writer.add_scalar("train/acc", train_acc, itr)
                writer.add_scalar("train/train_loss", train_loss, itr)


                info = nvmlDeviceGetMemoryInfo(handle)
                use = nvmlDeviceGetUtilizationRates(handle).gpu
                writer.add_scalar("GPU/mem_used",info.used/1e06,itr)
                writer.add_scalar("GPU/mem_utilization",info.used/info.total,itr)
                writer.add_scalar("GPU/gpu_utilization",use,itr)

        train_loss = (training_loss/training_samples)
        train_acc = (training_acc/training_samples)    
        writer.add_scalar("train/train_loss_e", train_loss, itr)   
        writer.add_scalar("train/train_acc_e", train_acc, itr)   
        
        val_loss = 0
        val_acc = 0
        val_samples = 0
        model.eval()
        for src, tgt in val_dataloader:
            src=src.cuda()
            tgt=tgt.cuda()

            logits = model(src)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt)

            with torch.no_grad():
                equals = (logits.sigmoid().argmax(1)==tgt).reshape(-1,1).detach().cpu()
                val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                val_loss += src.shape[0] * loss.item()
                val_samples += src.shape[0]

        val_loss = (val_loss/val_samples)
        val_acc = (val_acc/val_samples)    
        writer.add_scalar("val/val_loss_e", val_loss, itr)   
        writer.add_scalar("val/val_acc_e", val_acc, itr)
        
        model_dict = {"params":model.state_dict(),"itr":itr,"epoch":e}
        torch.save(model_dict,"linear.pth")