# -*- coding: utf-8 -*-

def getRepeatWord(from_file, to_file=None):
    '''
    transfer phoneme file to repeat words
    from_file: the full path of a phoneme.txt
    to_file: write repeat to.
    return:repeat words
    '''
    if to_file:
        fw = open(to_file, 'w+', encoding='utf-8-sig')
    f  = open(from_file, 'r', encoding='utf-8-sig')
    lines = f.readlines()
    f.close()
    preword = ''
    res = ''
    for i in range(1,len(lines)):
        line  =  lines[i]
        line  =  line.strip().replace('\n','')
        lst   =  line.split()
        start =  int(float(lst[0])*30)
        end   =  int(float(lst[1])*30)
        if len(lst) == 4:
            if lst[3] == '</s>':
                preword = '<s>'
            else:
                preword = lst[3]
            word = preword
        else:
            if lst[2] == 'sil':
                word = '<s>'
            else:
                word = preword

        for i in range(start + 1,end + 1):#[start+1,end]
            if to_file:
                fw.write(word + ' ')
            res = res + word + ' '
    if to_file:
        fw.close()
    return res

if __name__ == '__main__':
    import os
    data_root = r'C:\Users\sunyi03\Desktop\new_data'
    flst = os.listdir(data_root)
    f = open('./test_new_data.txt','w+', encoding='utf-8-sig')
    th = 1
    for name in flst:
        if name.find('phoneme.txt') < 0:continue
        wav = name.replace('sentence_phoneme.txt','16k.wav')

        res = getRepeatWord(r'%s\%s' % (data_root,name))
        f.write(res + '\n')
        os.system(r'xcopy %s\%s C:\Users\sunyi03\Desktop\FaceRecognitionaudio\Assets\new_data\%d.wav' % (data_root, wav, th))
        #os.system(r'xcopy C:\Users\sunyi03\Desktop\new_data\%s C:\Users\sunyi03\Desktop\FaceRecognitionaudio\Assets\zhurong\%s' % (wav,wav))
        th += 1
    f.close()