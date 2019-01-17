# -*- coding: utf-8 -*-
import os
import re 
import pdb
import random
import time
root_dir = 'C:/Users/sunyi03/Desktop/voice_cutting_new_eular'
dic     = {u'<pad>':0, u'<s>':1, u'<unk>':2}
dic_num = {u'<pad>':0, u'<s>':0, u'<unk>':0} 
class_dic = {}
index = 3
class_index = 0
dir_lst = os.listdir(root_dir)

fw= open('./data/one_hot_frame_train.txt', 'w+', encoding='utf-8-sig')
fwx= open('./word_frame_train.txt', 'w+', encoding='utf-8-sig')
flc = open('./data/labels_c.txt', 'w+', encoding='utf-8-sig')
flr = open('./data/labels_r.txt', 'w+', encoding='utf-8-sig')
errornum = 0
ferr = open('./errorlist.txt','w+', encoding='utf-8-sig')
def seperateData2ClassAndContinue(lines, class_dic, class_index):
    #import pdb
    #pdb.set_trace()
    #print("????:", len(class_dic))
    #seg = [[20,50,100] for i in range(51)]#每个划分成3个区间
    seg = [[5,20,40,70,100] for i in range(51)]#每个划分成5个区间
    for r in range(1):#数据扩增0倍
        for leftright in [[0,2,4,8,12,14,16,17]]:#[1,3,5,9,13,15,16,18]]: #左右数据扩增2倍
            for line in lines:
                tokenlst = line.split(' ')
                token_c = ''
                token_r = ''
                for i in leftright:
                    if i!=0 and i!=1:
                        token_c = token_c + '_'
                        token_r = token_r + '_'
                    random.seed(time.time())
                    #dt = random.random() * 10 - 5 #[-5,5]random float
                    dt = 0
                    after = max(min(float(tokenlst[i]) + dt, 100),0)
                    #print(dt, after)
                    token_r = token_r + str(after)
                    for j in range(len(seg[i])):
                        if after <= seg[i][j]:
                            token_c = token_c + str(j)
                            break
                if token_c not in class_dic:
                    class_dic[token_c] = class_index
                    w = class_index
                    class_index += 1
                else:
                    w = class_dic[token_c]
                flc.write(str(w) + ' ')
                flr.write(token_r + ' ')
            flc.write('\n')
            flr.write('\n')
    return class_index, 1 #数据扩增0倍

th = 1
for d in dir_lst:
    #pdb.set_trace()
    if d != '0918' and d!='0921':continue
    d = '%s/%s' % (root_dir, d)
    dlst = os.listdir(d)
    for dd in dlst:
        dd = '%s/%s' % (d,dd)
        flst = os.listdir(dd)
        for fname in flst:
            preword = ''
            if fname.find('phoneme') < 0:
                continue
            fname = '%s/%s' % (dd, fname)
            f = open(fname, 'r', encoding='utf-8-sig')
            lines = f.readlines()
            f.close()

            #f = open(fname, 'r', encoding='utf-8-sig')
            #blendshape = f.readlines()
            #f.close()

            f = open(fname.replace('phoneme','blendshape'), 'r', encoding='utf-8-sig')

            bslst = f.readlines()
            
            fnum = len(bslst) #real frame num
            f.close()

            print('file name:', fname)
            #if fname == r'C:/Users/sunyi03/Desktop/音素数据/qipa_004/17_1_phoneme.txt':
            #    pdb.set_trace()
            cot = 0
            tmpstr1 = ''
            tmpstr2 = ''
            is_error = 0
            for idx in range(len(lines)):
                #pdb.set_trace()
                #if idx == 84:pdb.set_trace()
                if idx == 0:continue
                lines[idx] = lines[idx].strip()
                lst = lines[idx].replace('</s>', '<s>').replace('sil','<s>').split()
                stime = int(float(lst[0])*30)
                etime = int(float(lst[1])*30)
                
                if len(lst) == 4:
                    word = lst[-1]
                    if word not in dic:
                        dic[word] = index
                        index += 1
                        dic_num[word] = 0
                    for i in range(stime+1, etime+1, 1):
                        #fw.write('%d ' % dic[word])
                        tmpstr1 = tmpstr1 + ' ' + str(dic[word])
                        #fwx.write('%s ' % word)
                        tmpstr2 = tmpstr2 + ' ' + word
                        dic_num[word] += 1
                    preword = word
                    if word != '<s>':
                        cot+=1
                else:
                    if lst[-1] == '<s>':
                        word = lst[-1]
                        for i in range(stime+1, etime+1, 1):
                            #fw.write('%d ' % dic[word])
                            tmpstr1 = tmpstr1 + ' ' + str(dic[word])
                            #fwx.write('%s ' % word)
                            tmpstr2 = tmpstr2 + ' ' + word
                            dic_num[word] += 1
                    else:
                        word = preword
                        for i in range(stime+1, etime+1, 1):
                            #fw.write('%d ' % dic[word])
                            #fwx.write('%s ' % word)
                            tmpstr1 = tmpstr1 + ' ' + str(dic[word])
                            #fwx.write('%s ' % word)
                            tmpstr2 = tmpstr2 + ' ' + word
                            dic_num[word] += 1
                
                if idx == len(lines) - 1:
                    if etime <= fnum:
                        word = preword
                        for i in range(etime+1,fnum+1):
                            #fw.write('%d ' % dic[word])
                            #fwx.write('%s ' % word)
                            tmpstr1 = tmpstr1 + ' ' + str(dic[word])
                            #fwx.write('%s ' % word)
                            #fw.write('%s' % dic[word])
                            #fwx.write('%s\n' % word)
                            tmpstr2 = tmpstr2 + ' ' + word
                            dic_num[word] += 1
                    elif etime > fnum:
                        print('error, etime : ', etime , ' fnum:', fnum)
                        is_error = 1

            #if fname == r'C:/Users/sunyi03/Desktop/音素数据/qipa_004/17_1_phoneme.txt':
            #    pdb.set_trace()
            #pdb.set_trace()
            f = open(fname.replace('phoneme','sentence'),'r')
            ccot = 0
            i = f.readline()
            i = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",i)
            for j in i:
                ccot += 1
            if abs(cot-ccot) > 2:
                #print('error. cot: ', cot, ' ccot:', ccot)
                errornum += 1
                ferr.write(fname + '\n')
                is_error = 1
            
            if not is_error:
                if fnum != len(tmpstr1.strip().split()):
                    print("blendshape: ", fnum, " wordlen:", len(tmpstr1.strip().split()))
                if tmpstr1.replace('\n','').strip() == '':
                    errornum += 1
                    ferr.write(fname + '\n')
                    continue
                class_index, times = seperateData2ClassAndContinue(bslst, class_dic, class_index)
                print('class_index:', class_index, ' times:', times)
                for t in range(int(times)):

                    import os
                    #import pdb
                    #pdb.set_trace()
                    #os.system(r'xcopy %s C:\Users\sunyi03\Desktop\LSTM\GT\%d.wav' % (fname.replace('_phoneme.txt','.wav').replace('/','\\'), th))
                    #os.system(r'xcopy %s C:\Users\sunyi03\Desktop\LSTM\GT\%d_blendshape_gt.txt' % (fname.replace('phoneme','blendshape').replace('/','\\'), th))
                    fw.write('%s\n' % tmpstr1)
                    fwx.write('%s\n' % tmpstr2)
                    th += 1
            
ferr.close()
print('error num : ', errornum)
f = open('word_frame_num.txt','w+', encoding='utf-8-sig')
f.write(str(dic))
f.close()

f = open('word2index.txt','w+', encoding='utf-8-sig')
f.write(str(dic_num))
f.close()

f = open('c2oric.txt','w+', encoding='utf-8-sig')
f.write(str(class_dic))
f.close()
#print(os.getcwd())
'''
os.chdir('./data')
train_rate = 0.9
fdata  = open('all_data.txt', 'r',encoding='utf-8-sig')
flabel = open('all_label.txt','r',encoding='utf-8-sig')
data_lines  = fdata.readlines()
label_lines = flabel.readlines()
rangemax = [0 for i in range(51)] 
for line in label_lines:
    tokenlst = line.split(' ')
    for token in tokenlst:
        lst = token.split('_')
        for i in range(len(lst)):
            rangemax[i] = max(rangemax[i], float(lst[i]))

f = open('rangemax.txt','w+',encoding='utf-8-sig')
for i in range(51):
    f.write(str(rangemax[i]) + '\n')
f.close()
fdata.close()
flabel.close()

ftrain_data  = open('train_data.txt','w+',encoding='utf-8-sig')
fval_data    = open('val_data.txt','w+',encoding='utf-8-sig')
ftrain_label = open('train_label.txt','w+',encoding='utf-8-sig')
fval_label   = open('val_label.txt','w+',encoding='utf-8-sig')

for i in range(int(len(data_lines) * train_rate)):
    ftrain_data.write(data_lines[i])
    ftrain_label.write(label_lines[i])

for i in range(int(len(data_lines) * train_rate), len(data_lines)):
    fval_data.write(data_lines[i])
    fval_label.write(label_lines[i])

ftrain_data.close()
fval_data.close()
ftrain_label.close()
fval_label.close()
'''