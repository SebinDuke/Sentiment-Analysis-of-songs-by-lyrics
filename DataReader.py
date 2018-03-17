import re

#function to read one folder to 2d array
def readData(path,emotion):
    Data=[]
    lyrics=[]
    for i in range(1,107):
        filename=path+emotion+"_"+str(i)+".txt"
        try:
            with open(filename,'r',encoding='latin-1') as file:
                lyric=file.read()
                (lyric,count)=re.subn(r"(\n|\t)","",lyric)
                (lyric,count)=re.subn(r"(\[\d\d:\d\d\.\d\d\])","",lyric)
                lyrics.append(lyric)
        except:
            break


    #with codecs.open(path+"info.txt", mode='r', errors='ignore') as file:
    with open(path+"info.txt",'r',encoding='latin-1') as file:
        i=0
        for line in file:
            row=line.split(':')
            row=[i.strip() for i in row]
            #row.append(emotion)
            row.append(lyrics[i])
            Data.append(row)
            i+=1

    return Data

"""
Data=readData("Data-Set/Angry/Train/","angry")
for i in Data:
    print(i)
"""
