import DataReaderHindi as DRH

Ang_Songs=DRH.readData("Hindi_Dataset/Angry/",'A')
Sad_Songs=DRH.readData("Hindi_Dataset/Sad/",'S')
Rel_Songs=DRH.readData("Hindi_Dataset/Relaxed/",'R')
SongsTrain=[Ang_Songs, Sad_Songs, Rel_Songs]

SongsWordsTrain=[[],[]]
for i in range(3):
	for song in SongsTrain[i]:
		s=song[5]
		SongsWordsTrain[0].append(s)
		SongsWordsTrain[1].append(i)

print(SongsWordsTrain[0])
