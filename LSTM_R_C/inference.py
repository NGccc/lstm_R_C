import torch
import numpy as np
from utils import getRepeatWord

def getmodel():
    import pdb
    pdb.set_trace()
    model = torch.load('./eye_brow_model.pth')
    return model.cuda()

def deal_data(txt):
    f = open('./word_frame_num.txt', 'r', encoding='utf-8-sig')
    x = f.read()
    dic = eval(x)
    
    lst = txt.strip().replace('\n','').split()
    rs  = []
    for i in lst:
        if str(i) not in dic:
            rs.append(2) #<unk>
            #print('str(%s) not in dic' % str(i))
        else:
            rs.append((dic[str(i)]))
    rs = torch.LongTensor(np.array(rs).astype('long')).unsqueeze(0)
    return rs, torch.LongTensor(np.array([rs.shape[1]]))

def getFaceActionFromTxt(model, txt):
    data = deal_data(txt)
    import pdb
    pdb.set_trace()
    model.eval()
    data, lengths = data
    data = data.long().cuda()

    _, out = model(data, lengths)
    print(out)
    selected_left  = [(i+3) for i in [0,2,4,8,12,14,16,17]]
    selected_right = [(i+3) for i in [1,3,5,9,13,15,16,18]]
    length = lengths[0].data
    out_bs = np.zeros((length, 54))
    out_bs[:,selected_left]  = out[0,:length,:].data.cpu().numpy() * 100.0
    out_bs[:,selected_right] = out[0,:length,:].data.cpu().numpy() * 100.0
    return out_bs

def getFaceActionFromPhoneme(model, txt):
    '''
    getRepeatWord(r'%s\%s' % (data_root, name))
    data = deal_data(txt)
    
    model.eval()
    data, lengths = data
    data = data.long().cuda()

    _, out = model(data, lengths)
    print(out)
    return out
    '''
    pass


'''
if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    model = getmodel()
    out = getFaceActionFromTxt(model, u'<s> <s> <s> <s> <s> <s> 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 嘿 <s> <s> <s> <s> <s> 京 京 京 京 京 京 京 城 城 城 城 城 城 里 里 里 这 这 这 么 么 么 么 多 多 多 多 多 人 人 人 人 人 人 人 人 人 <s> <s> 我 我 我 我 我 我 我 哪 哪 哪 哪 能 能 能 能 能 能 全 全 全 全 全 全 全 全 认 认 认 认 得 得 得 得 得 得 得 <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> 反 反 反 反 反 反 反 反 正 正 正 正 人 人 人 人 家 家 给 给 给 给 给 钱 钱 钱 钱 钱 钱 钱 钱 钱 钱 钱 钱 我 我 我 我 我 们 们 们 办 办 办 办 办 办 事 事 事 事 事 事 事 就 就 就 行 行 行 行 了 了 了 了 了 了 <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> 喏 喏 喏 喏 喏 喏 喏 喏 喏 <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> 这 这 这 这 这 这 这 是 是 是 您 您 您 的 的 马 马 马 马 马 马 马 马 马 <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> <s> ')
    np.savetxt('out.txt',out)
    print(out)
'''