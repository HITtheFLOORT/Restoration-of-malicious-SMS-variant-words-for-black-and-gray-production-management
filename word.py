import csv
from zhtools.langconv import *
def Traditional2Simplified(sentence):
  '''
  将sentence中的繁体字转为简体字
  :param sentence: 待转换的句子
  :return: 将句子中繁体字转换为简体字之后的句子
  '''
  sentence = Converter('zh-hans').convert(sentence)
  return sentence
f=open("train_public.csv","r",encoding="utf8",errors='ignore')
a=csv.reader(f)
index=0
import csv
csvfile=open('new.csv','w',newline="",errors='ignore')
writer = csv.writer(csvfile)
writer.writerow(["source","src_sentences","tar_sentences"])
for i in a:
    index+=1
    strs=str(i).split("\\t")
    s1=strs[1].replace("'","").replace(" ","").replace("（","(").replace("）",")").replace("【","").replace("】","").replace("?","")
    s2=strs[2].replace(']','').replace("'","")
    writer.writerow(["web",s1,s2])

f.close()
csvfile.close()