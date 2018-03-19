import re

#function to read one folder to 2d array
def readData(path,emotion):
    Data=[]
    lyrics=[]
    for i in range(1,6):
        filename=path+emotion+str(i)+"_E"
        try:
            with open(filename,'r') as file:
                lyric=file.read()
                #(lyric,count)=re.subn(r"(\\n|\\u....|\t)","",lyric)
                #(lyric,count)=re.subn(r"(\[\d\d:\d\d\.\d\d\])","",lyric)
                lyrics.append(lyric)
        except:
            break


    #with codecs.open(path+"info.txt", mode='r', errors='ignore') as file:
    with open(path+"info",'r') as file:
        i=0
        for line in file:
            row=line.split(':')
            row=[i.strip() for i in row]
            row.append(lyrics[i])
            Data.append(row)
            i+=1

    return Data
"""
Data=readData("Hindi_Dataset/Angry/",'A')
for i in Data:
    print(i)
"""