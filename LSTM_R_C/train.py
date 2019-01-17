# -*- coding: utf-8 -*-
import config
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from BiLSTM_Attention_model import BLSTMA
from mydataset import MyDataset

from tensorboardX import SummaryWriter
import os
import pdb
import math
import utils

def hook(module, input, output):
    print(output[0].shape)
    print(output[0].data)

opts = config.opts
model = BLSTMA()

# ==== tensorboard ====
writer = SummaryWriter(log_dir='logs_adam')

batch_size      = opts.batch_size
pre_model_path  = opts.model
epochs          = opts.epochs
gpu             = opts.gpu
lr              = opts.lr

if pre_model_path:
    if os.path.exists(pre_model_path):
        model.load_state_dict(torch.load(pre_model_path))
    else:
        print('error.check ur model path')
        exit()

if gpu >=0 :
    model = model.cuda()

#import pdb
#pdb.set_trace()
torch.save(model,'./model/eye_brow_model.pth')

#import pdb
#pdb.set_trace()
#xmodel = torch.load('./model/eye_brow_model.pth')

#grad hook
#for param in model.parameters():
#    print(param)
#model.lstmF.register_backward_hook(hook)
#model.FC_r.register_backward_hook(hook)

loss_func_r = nn.L1Loss(reduce=False)
loss_func_c = nn.CrossEntropyLoss(reduce=False)

optimizer = optim.Adam(model.parameters(), lr=lr)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)

#train_dataset = MyDataset(train=True, transform=transforms.ToTensor())
#train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


#when inference 
batch_size  = 1
epochs      = 1
th          = 1

val_dataset = MyDataset(train=False, transform=transforms.ToTensor())
val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#when inference 
dloader  = [val_loader, ]
is_train = 0

#when train
#dloader  = [train_loader, ]
#is_train = 1

tag = ['val', 'train']
for epoch in range(epochs):
    #recode running_loss[1] is train, 0 is val
    if (epoch + 1) % 100 == 0:
        lr = lr / 3.0
    running_loss_r = torch.tensor([0.0, 0.0])
    running_loss_c = torch.tensor([0.0, 0.0])
    running_acc = torch.tensor([0.0, 0.0])
    running_total_len = torch.tensor([0.0, 0.0])
    if gpu:
        running_loss_r, running_loss_c, running_acc, running_total_len = running_loss_r.cuda(), running_loss_c.cuda(), running_acc.cuda(), running_total_len.cuda()
    step = [1,1]
    print('epoch:[%d/%d]' % (epoch, epochs))
    for dataloader in dloader:
        if is_train:
            model.train()
        else:
            model.eval()
        for i, data in enumerate(dataloader, 1):
            if is_train:
                #import pdb
                #pdb.set_trace()
                data, labelr, labelc, lengths = data
                data, labelr, labelc = data.long(), labelr.float(), labelc.long()

                if gpu >=0:
                    data, labelr, labelc = data.cuda(), labelr.cuda(), labelc.cuda()
            else:
                data, lengths = data
                data = data.long()
                print("?????")
                if gpu >=0:
                    data = data.cuda()
            
            lengths, indices = torch.sort(lengths, descending=True)
            data  = data[indices]
            out_c, out_r = model(data, lengths)
            if not is_train:
                print('test[%d]: length:%d' % (th,lengths[0]))
                selected_left  = [(i+3) for i in [0,2,4,8,12,14,16,17]]
                selected_right = [(i+3) for i in [1,3,5,9,13,15,16,18]]
                
                for length in lengths:
                    out_bs = np.zeros((length, 54))
                    out_bs[:,selected_left]  = out_r[0,:length,:].data.cpu().numpy() * 100.0
                    out_bs[:,selected_right] = out_r[0,:length,:].data.cpu().numpy() * 100.0
                    np.savetxt(r'C:\Users\sunyi03\Desktop\FaceRecognitionaudio\Assets\long_data\%d_blendshape_fake.txt' % th, out_bs)
                    th += 1
                #exit()
            else:
                mask = torch.zeros((out_c.shape[0],out_c.shape[1])).float()
                sentence_len = lengths.sum().float()
                if opts.gpu:
                    sentence_len = sentence_len.cuda()
                    mask = mask.cuda()
                
                th = 0
                for length in lengths:
                    mask[th,:length] = 1
                    th += 1

                labelr = labelr[indices]
                labelc = labelc[indices]
                _, acc = torch.max(out_c, 2)
                labelc = labelc[:,:acc.shape[1]]
                labelr = labelr[:,:acc.shape[1]]
                acc = ((acc == labelc).float() * mask).sum()
                alpha = 0.5
                
                loss_c = (loss_func_c(out_c.reshape((out_c.shape[0]*out_c.shape[1],-1)), labelc.reshape(-1)).reshape(out_c.shape[0],-1) * mask).sum()
                loss_r = (loss_func_r(out_r.reshape((out_r.shape[0]*out_r.shape[1],-1)), labelr.reshape(labelr.shape[0]*labelr.shape[1],-1)).reshape(labelr.shape[0],labelr.shape[1],-1).sum(2) * mask).sum()
                loss = (alpha * loss_c + (1 - alpha) * loss_r)
                running_loss_r[is_train] += loss_r
                running_loss_c[is_train] += loss_c
                running_total_len[is_train] += sentence_len
                running_acc[is_train] += acc

                #log and print
                print('[%s]step %d: class_loss:%f regression_loss:%f loss:%f| acc:%f%% lr:%f' % (tag[is_train], step[is_train], loss_c / sentence_len, loss_r / sentence_len, loss / sentence_len, 100.0 * acc / sentence_len, lr))
                writer.add_scalar('data/[%s]_regression_loss' % tag[is_train], loss_r / sentence_len, step[is_train])
                writer.add_scalar('data/[%s]_class_loss' % tag[is_train], loss_c / sentence_len, step[is_train])
                writer.add_scalar('data/[%s]_loss' % tag[is_train], loss / sentence_len, step[is_train])
                writer.add_scalar('data/[%s]_class_acc' % tag[is_train], 100.0 * acc / sentence_len, step[is_train])

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                step[is_train] += 1
        
        #save model
        if is_train and (epoch + 1) % opts.save_frequence == 0:
            print('[%s]epoch [%d/%d] finish, lossr:%f lossc:%f loss:%f | acc:%f %%' % (tag[is_train],
            epoch, epochs, running_loss_r[is_train] / running_total_len[is_train].float(), 
            running_loss_c[is_train] / running_total_len[is_train].float(), 
            (alpha * running_loss_c[is_train] + (1 - alpha) * running_loss_r[is_train]) / running_total_len[is_train].float(),
            100.0 * (running_acc[is_train] / running_total_len[is_train].float()
            )))

            torch.save(model.state_dict(), 
            '{}/epoch_{}_acc_{:.2f}_lossr_{:.2f}_lossc_{:.2f}_loss_{:.2f}.pth'.format(
            opts.save_model,
            epoch+1,
            running_acc[is_train].float() / running_total_len[is_train].float() * 100.0,
            running_loss_r[is_train] / running_total_len[is_train].float(), 
            running_loss_c[is_train] / running_total_len[is_train].float(), 
            (alpha * running_loss_c[is_train] + (1 - alpha) * running_loss_r[is_train]) / running_total_len[is_train].float(), 
        ))
        #is_train = 1 - is_train  