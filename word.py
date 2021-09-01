import csv
f=open("train_public.csv","r",encoding="utf8",errors='ignore')
a=csv.reader(f)
index=0
for i in a:
    index+=1
    strs=str(i).split("\\t")
    print(strs[1].replace("'","").replace(" ","").replace("（","(").replace("）",")"))
    print(strs[2].replace(']','').replace("'",""))
    if index == 100:
        break

