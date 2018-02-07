import re

angry=[]
lyrics=[]
path="Data-Set/Angry/Train/"
for i in range(1,101):
    filename=path+"angry_"+str(i)+".txt"
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
        angry.append(row)


for i in angry:
    print(i)