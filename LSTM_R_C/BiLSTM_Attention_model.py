# -*- coding: utf-8 -*-
import torch
from torch import nn
import config
import numpy as np
import pdb
import torch.nn.functional as F

def reverse_padded_sequence(inputs, lengths):
    '''
    [1,2,3,4,0,0]
    [1,2,4,0,0,0]
    
    [4,3,2,1,0,0]
    [4,2,1,0,0,0]
    inputs:
    [batch_size, sentence_len, embedding_len]
    '''
    th = 0
    maxlen = lengths[0]
    emddim = inputs.shape[2]
    flip = torch.zeros((inputs.shape[0], lengths[0], inputs.shape[2]))
    for length in lengths:
        idxf = [i for i in range(length)]
        idxf.reverse()
        idx = idxf + [i for i in range(length,maxlen)]
        flip[th,:,:] = inputs[th,idx,:] 
        th += 1
    if opts.gpu:
        flip = flip.cuda()
    return flip


#get hyper parameters
opts = config.opts
class BLSTMA(nn.Module):
    def __init__(self, gpu=False):
        super(BLSTMA, self).__init__()
        self.in_dim  = opts.in_dim
        self.n_layer = opts.n_layer
        self.h_dim   = opts.h_dim
        self.gpu     = opts.gpu
        self.key_num = opts.key_num  #关键点个数
        self.class_num = opts.class_num
        self.embeds = nn.Embedding(opts.vocab_size, self.in_dim)
        self.lstmF   = nn.LSTM(self.in_dim, self.h_dim, self.n_layer, batch_first=True, dropout=0.3)
        self.lstmB   = nn.LSTM(self.in_dim, self.h_dim, self.n_layer, batch_first=True)
        
        self.attention = self.AttentionLayer
        
        #self.bn1        = nn.BatchNorm1d(self.h_dim * 2, momentum=0.5)
        self.FC_c        = nn.Linear(self.h_dim, self.class_num) #h_dim = c_dim
        
        #self.relu       = nn.ReLU()
        #self.FC2        = nn.Linear(self.h_dim, self.key_num)
        self.FC_r        = nn.Linear(self.h_dim, self.key_num)
        self.Sig         = nn.Sigmoid() #[0,1]

    def init_hidden(self, cur_batch_size):
        hidden_a = torch.randn(self.n_layer, cur_batch_size, self.h_dim)
        hidden_b = torch.randn(self.n_layer, cur_batch_size, self.h_dim)
        if self.gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()
        return (hidden_a, hidden_b)

    def forward(self, x, ori_length_lst):
        #import pdb
        #pdb.set_trace()
        self.hidden1 = self.init_hidden(x.shape[0])
        embedding_x = self.embeds(x)
        
        packed_x = nn.utils.rnn.pack_padded_sequence(embedding_x, ori_length_lst.numpy().tolist(), batch_first=True) 
        x, self.hidden1 = self.lstmF(packed_x, self.hidden1) #全0的隐藏层省略输出
        x,_ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) #补充原来全0的隐藏层的补齐
        
        out1 = self.FC_c(x)
        out2 = self.Sig(self.FC_r(x))
        return out1, out2
        
        '''
        flip_x = reverse_padded_sequence(embedding_x, ori_length_lst)
        
        self.hidden2 = self.init_hidden(flip_x.shape[0])
        packed_flip_x = nn.utils.rnn.pack_padded_sequence(flip_x, ori_length_lst.numpy().tolist(), batch_first=True) 
        x1, self.hidden2 = self.lstmB(packed_flip_x, self.hidden2)
        x1,_ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True)

        #x1.shape = [batch_size, sentence_len, h_dim]
        x1 = reverse_padded_sequence(x1, ori_length_lst)
        H = torch.cat([x, x1], 2)
        #BiLSTM output : H
        #C = self.attention(H, ori_length_lst)
        
        out = self.relu(self.FC1(H))
        out = self.FC2(out)
        return out
        '''

    def AttentionLayer(self, H, lengths): 
        th = 0
        #import pdb
        #pdb.set_trace()
        C = torch.zeros(H.shape).float()
        if opts.gpu:
            C = C.cuda()
        for length in lengths:
            x = H[th,:length,:]
            xy = x.mm(x.t())
            xmod = x.mul(x).sum(1).sqrt().unsqueeze(1)
            xymod = xmod.mm(xmod.t())
            cos_similary = xy / xymod

            a = F.softmax(cos_similary, 1)
            
            for i in range(length):
                C[th,i,:] = (a[i:i+1,:].t() * x).sum(0)
            th += 1
        return C
            

if __name__ == '__main__':
    loss_func = nn.MSELoss(reduce=False)
    #lst = torch.tensor([4,3]) #packed x.shape = [4+3,in_dim]
    #对应的补0的部分会被记录下来，然后经过LSTM之后，这些位置的隐藏层h为全0向量
    #x = torch.tensor([[1,2,3,4],[1,2,5,0]]).cuda() 
    import pdb
    pdb.set_trace()
    model = BLSTMA()
    model.load_state_dict(torch.load('./model/epoch_6_acc_2.97_loss_0.23.pth'))
    model = model.cuda()

    data = np.loadtxt('error_data.txt')
    lengths = np.loadtxt('error_lengths.txt')
    data = torch.from_numpy(data).cuda().long()
    lengths = torch.from_numpy(lengths).long()
   
    out = model(data,lengths)
    print(out)