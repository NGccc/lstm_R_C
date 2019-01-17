# -*- coding: utf-8 -*-
import numpy as np
import config
from torch.utils import data
import pdb
opts = config.opts

class MyDataset(data.Dataset):
    def __init__(self, train=True, transform=None):
        #import pdb
        #pdb.set_trace()
        self.data    = opts.data
        self.train   = train
        self.maxlen  = opts.maxlength
        self.key_num = opts.key_num
        if self.train:
            with open('%s/one_hot_frame_train.txt' % self.data, 'r', encoding='utf-8-sig') as f:
                self.train_data = f.readlines()
            with open('%s/labels_c.txt' % self.data, 'r', encoding='utf-8-sig') as f:
                self.train_labelc = f.readlines()
            with open('%s/labels_r.txt' % self.data, 'r', encoding='utf-8-sig') as f:
                self.train_labelr = f.readlines()
            self.train_length = len(self.train_data)
        else:
            with open('%s/test_new_data.txt' % self.data, 'r', encoding='utf-8-sig') as f:
                self.val_data = f.readlines()
            f = open('./word_frame_num.txt', 'r', encoding='utf-8-sig')
            x = f.read()
            dic = eval(x)
            rs = []
            for line in self.val_data:
                lst=line.strip().replace('\n','').split()
                res = ''
                for i in lst:
                    if str(i) not in dic:
                        res=res+'2 ' #<unk>
                        print('str(%s) not in dic' % str(i))
                    else:
                        res=res+str(dic[str(i)])+' '
                res=res+'\n'
                rs.append(res)
            self.val_data = rs
            '''
            with open('%s/labels_c_val.txt' % self.data, 'r', encoding='utf-8-sig') as f:
                self.val_labelc = f.readlines()
            with open('%s/labels_r_val.txt' % self.data, 'r', encoding='utf-8-sig') as f:
                self.val_labelr = f.readlines()
            '''
            self.val_length = len(self.val_data)
            #import pdb
            #pdb.set_trace()

    def __getitem__(self, index):
        if self.train:
            data_lst  = self.train_data
            labelc_lst = self.train_labelc
            labelr_lst = self.train_labelr
        else:
            data_lst  = self.val_data
            data = data_lst[index].replace('\n','').strip().split(' ')
            data = np.array(data).astype('long')
            data1 = np.zeros((self.maxlen, ))
            data1[:data.shape[0]] = data
            return data1, data.shape[0]
        
        data = data_lst[index].replace('\n','').strip().split(' ')
        #print('??',data_lst[index],'??')
        data = np.array(data).astype('long')
        
        labelr = labelr_lst[index].replace('\n','').strip().split(' ')
        labelc = labelc_lst[index].replace('\n','').strip().split(' ')
        labelr_lst = []
        labelc_lst = []
        for la in labelr:
            labelr_lst.append(la.split('_'))
        #for la in labelc:
        #    labelc_lst.append(la.split('_'))
        label_1 = np.array(labelr_lst).astype('float') / 100.0 #[0,100] to [0,1]
        label_2 = np.array(labelc).astype('long')
        data1 = np.zeros((self.maxlen, ))
        data1[:data.shape[0]] = data
        
        labelr = np.zeros((self.maxlen, self.key_num))
        labelc = np.zeros((self.maxlen))
        labelr[:data.shape[0],:] = label_1
        labelc[:data.shape[0]] = label_2
        #print(index)
        #(161,) (161,3) 66
        return data1, labelr, labelc, data.shape[0]

    def __len__(self):
        if self.train:
            return self.train_length
        else:
            return self.val_length