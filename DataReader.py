import re

def readData(path,emotion):
    Data=[]
    lyrics=[]
    for i in range(1,101):
        filename=path+emotion+"_"+str(i)+".txt"
        try:
            with open(filename,'r',encoding='latin-1') as file:
                lyric=file.read()
                (lyric,count)=re.subn(r"(\\n|\\u....|\t)","",lyric)
                (lyric,count)=re.subn(r"(\[\d\d:\d\d\.\d\d\])","",lyric)
                lyrics.append(lyric)
        except:
            break


    #with codecs.open(path+"info.txt", mode='r', errors='ignore') as file:
    with open(path+"info.txt",'r',encoding='latin-1') as file:
        i=0
        for line in file:
            row=line.split(':')
            row.append(lyrics[i])
            Data.append(row)
            i+=1

    return Data

"""
Data=readData("Data-Set/Angry/Train/","angry")
for i in Data:
    print(i)
"""