import re

def ReadData(path,emotion):
    Data=[]
    lyrics=[]
    for i in range(1,101):
        filename=path+emotion+"_"+str(i)+".txt"
        with open(filename,'r') as file:
            lyric=file.read()
            (lyric,count)=re.subn(r"(\\n|\\u....|\t)","",lyric)
            (lyric,count)=re.subn(r"(\[\d\d:\d\d\.\d\d\])","",lyric)
            lyrics.append(lyric)


    with open(path+"info.txt",'r') as file:
        i=0
        for line in file:
            row=line.split(':')
            row.append(lyrics[i])
            Data.append(row)
            i+=1

    return Data

Data=ReadData("Data-Set/Angry/Train/","angry")
for i in Data:
    print(i)