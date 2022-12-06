from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from src.model import accident
from tqdm import tqdm
import os
import numpy as np
from src.eval_tools import evaluation, print_results, vis_results,evaluate_earliness
from src.bert import opt
from src.dataset import DADA
from natsort import natsorted
os.environ['CUDA_VISIBLE_DEVICES']= '0'
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

# device = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
num_epochs = 50
# learning_rate = 0.0001
batch_size =2
shuffle = True
pin_memory = True
num_workers = 1
rootpath=r''
frame_interval=1
input_shape=[224,224]
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
val_data=DADA(rootpath , 'testing', interval=1,transform=transform)
valdata_loader=DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True,drop_last=True)
def write_scalars(logger, epoch, loss):
    logger.add_scalars('train/loss',{'loss':loss}, epoch)

def write_test_scalars(logger, epoch, losses, metrics):
    # logger.add_scalars('test/loss',{'loss':loss}, epoch)
    logger.add_scalars('test/losses/total_loss',{'Loss': losses}, epoch)
    logger.add_scalars('test/accuracy/AP',{'AP':metrics['AP'], 'PR80':metrics['PR80']}, epoch)
    logger.add_scalars('test/accuracy/time-to-accident',{'mTTA':metrics['mTTA'], 'TTA_R80':metrics['TTA_R80']}, epoch)

def test(test_dataloader, model):
    all_pred = []
    all_labels = []
    losses_all = []
    all_toas = []
    model.eval()
    with torch.no_grad():
        loop = tqdm(test_dataloader,total = len(test_dataloader), leave = True)
        for imgs,focus,info,label,texts in loop:
            # torch.cuda.empty_cache()
            imgs=imgs.to(device)
            focus=focus.to(device)
            labels = label
            toa = info[0:, 4].to(device)
            labels = np.array(labels).astype(int)
            labels = torch.from_numpy(labels)
            labels = labels.to(device)
            outputs = model(imgs,focus,labels.long(),toa,texts)
            num_frames = imgs.size()[1]
            batch_size = imgs.size()[0]
            pred_frames = np.zeros((batch_size,num_frames),dtype=np.float32)
            for t in range(num_frames):
                pred = outputs[t]
                # print( pred)
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            #gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(toa.cpu().numpy()).astype(np.int64)
            all_toas.append(toas)
            loop.set_postfix(val_loss = sum(losses_all))
    all_pred = np.concatenate(all_pred)
    all_labels = np.concatenate(all_labels)
    all_toas = np.concatenate(all_toas)
    return all_pred, all_labels, all_toas

def test_data():
    h_dim = 256
    n_layers = 1
    depth=4
    adim=opt.adim
    heads=opt.heads
    num_tokens=opt.tokens
    c_dim=opt.c_dim
    s_dim1=opt.s_dim1
    s_dim2=opt.s_dim2
    keral=opt.keral
    num_class=opt.num_class
    ckpt_path = r''
    weight = torch.load(ckpt_path)
    model=accident(h_dim,n_layers,depth,adim,heads,num_tokens,c_dim,s_dim1,s_dim2,keral,num_class).to(device)
    model.eval()
    model.load_state_dict(weight)
    print('------Starting evaluation------')
    all_pred, all_labels, all_toas= test(valdata_loader,model)
    mTTA = evaluate_earliness(all_pred, all_labels, all_toas, fps=30, thresh=0.5)
    print("\n[Earliness] mTTA@0.5 = %.4f seconds." % (mTTA))
    AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, all_toas, fps=30)
    print("[Correctness] AP = %.4f, mTTA = %.4f, TTA_R80 = %.4f" % (AP, mTTA, TTA_R80))
    all_vid_scores=[max(pred[int(toa):]) for toa, pred in zip(all_toas, all_pred)]
    AUC=roc_auc_score(all_labels,all_vid_scores)
    print("[Correctness] v-AUC = %.5f." % (AUC))

if __name__=="__main__":
    test_data()