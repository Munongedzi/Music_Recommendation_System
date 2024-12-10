import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class SongInfo:
    def __init__(self, track, dance, energy, key, loud, speech, acou, instr, live, valence, tempo, dur, num):
        self.track = track  # string
        self.dance = dance  # int
        self.energy = energy  # int
        self.key = key  # string
        self.loud = loud  # int
        self.speech = speech  # int
        self.acou = acou  # int
        self.instr = instr  # int
        self.live = live  # int
        self.valence = valence  # int
        self.tempo = tempo  # int
        self.dur = dur  # int
        self.num = num  # int

    def __str__(self):
        return (f"Track: {self.track}, Danceability: {self.dance}, Energy: {self.energy}, "f"Key: {self.key}, Loudness: {self.loud}, Speechiness: {self.speech}, "f"Acousticness: {self.acou}, Instrumentalness: {self.instr}, "f"Liveness: {self.live}, Valence: {self.valence}, Tempo: {self.tempo}, "f"Duration (ms): {self.dur}, Number: {self.num}" )


def printSong(song):
    sp = ","
    #song.num,
    print(song.num, sp, song.track, sp, song.dance, sp, song.energy, sp, song.key, sp, song.loud, sp, song.speech, sp, song.acou, sp, song.instr, sp, song.live, sp, song.valence, sp, song.tempo, sp, song.dur  )

def printTitle(song):
    print(song.track)

def printTargSong(song):
    sp = ","
    t = 'Target Song:'
    print( t, song.num, sp, song.track, sp, song.dance, sp, song.energy, sp, song.key, sp, song.loud, sp, song.speech, sp, song.acou, sp, song.instr, sp, song.live, sp, song.valence, sp, song.tempo, sp, song.dur  )


def pullData(filepath):
    columns_to_extract = ["Track", "Danceability", "Energy", "Key", "Loudness", "Speechiness", "Acousticness",
                          "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms"]

    data = pd.read_csv(filepath, header=0)
    selected_data = data[columns_to_extract]
    vector_data = selected_data.values.tolist()
    allSongs = []

    counter = 0
    for song in vector_data:
        track = song[0]
        dance = song[1]
        energy = song[2]
        key = song[3]
        loud = song[4]
        speech = song[5]
        acou = song[6]
        instr = song[7]
        live = song[8]
        valence = song[9]
        tempo = song[10]
        dur = song[11]
        num = counter

        counter += 1
        dur = dur / 60000  #change song from ms to minutes
        new_song = SongInfo(track, dance, energy, key, loud, speech, acou, instr, live, valence, tempo, dur, num)
        allSongs.append(new_song)

    return allSongs

def collabFilter(allSongs, allUsers, targetUser, numRecs):

    allSongNum = []
    for song in allSongs:
        allSongNum.append(song.num)

    userItemMatrix = np.zeros((len(allUsers), len(allSongs)))

    for user_idx, user_songs in enumerate(allUsers):
        for song in user_songs:

            col_idx = allSongNum.index(song.num)
            userItemMatrix[user_idx, col_idx] = 1

    userSongs = allUsers[targetUser]

    #df = pd.DataFrame(userItemMatrix, columns=allSongNum)
    userItemMatrix = normalize(userItemMatrix, axis=1)
    user_similarity = cosine_similarity(userItemMatrix)
    #print("User Similarity Matrix:")
    #print(user_similarity)

    targetMatrix = user_similarity[targetUser]
    #print("target matrix")
    #print(targetMatrix)

    closestUsers = []
    numOfUsers = len(allUsers)

    for i in range( numOfUsers ):
        if (i == targetUser):
            continue

        if (targetMatrix[i] > 0):
            x = [i, targetMatrix[i]]
            closestUsers.append(x)

    closestUsers = sorted(closestUsers, key=lambda x: x[1], reverse=True)
    #print("Closest Users")
    #print(closestUsers)

    closeSongs = []


    for i in closestUsers:
        for song in allUsers[i[0]]:
            closeSongs.append(song)


    for song1 in closeSongs:
        for song2 in userSongs:
            if song1.track == song2.track or song1.num == song2.num:
                closeSongs.remove(song1)

    if ( len(closeSongs) == 0):
        print("Wow it looks like this user has such unique taste that no other users on the platform are even close! You should try the Content Filter Recommendation Service instead! ")

    return closeSongs[:numRecs]


def contentFilter(allSongs, targetSongNum, numRecs):
    scalar = MinMaxScaler()

    numeric_features = []
    valid_songs = []

    for song in allSongs:
        try:
            features = [float(song.dance), float(song.energy), float(song.loud),
                        float(song.speech), float(song.acou), float(song.instr),
                        float(song.live), float(song.valence), float(song.tempo),
                        float(song.dur)]
            if all(np.isfinite(features)):
                numeric_features.append(features)
                valid_songs.append(song)
        except (ValueError, TypeError):
            continue

    scaled_features = scalar.fit_transform(numeric_features)

    similarity_matrix = cosine_similarity(scaled_features)

    target_index = next((i for i, song in enumerate(valid_songs) if song.num == targetSongNum), None)
    if target_index is None:
        raise ValueError(f"Target song with num {targetSongNum} not found in valid songs.")

    similar_songs = similarity_matrix[target_index]

    similar_indices = np.argsort(-similar_songs)[1:numRecs+1]

    recommended_songs = [valid_songs[idx] for idx in similar_indices]

    for song1 in recommended_songs:
        for song2 in recommended_songs:
            if(song1.track == song2.track):
                recommended_songs.remove(song2)

    return recommended_songs

#print("Starting")
songPool = pullData("E:\\Algs\\MusicRecommender\\data\\spotify1.csv")
#print("Pulled Data from csv file")

# this charming man : 12530
# The Less I Know The Better : 13549
# Fluorescent Adolescent : 11974

num_of_recs = 50
t1 = 6039
t2 = 12530
t3 = 8139
t4 = 14380
t5 = 4875
t6 = 5199
t7 = 11425
t8 = 18341
t9 = 2556
t10 = 11974

u1 = contentFilter(songPool, t1, num_of_recs)
jenny = contentFilter(songPool, t2, num_of_recs)
u3 = contentFilter(songPool, t3, num_of_recs)
u4 = contentFilter(songPool, t4, num_of_recs)
u5 = contentFilter(songPool, t5, num_of_recs)
u6 = contentFilter(songPool, t6, num_of_recs)
u7 = contentFilter(songPool, t7, num_of_recs)
u8 = contentFilter(songPool, t8, num_of_recs)
u9 = contentFilter(songPool, t9, num_of_recs)
u10 = contentFilter(songPool, t10, num_of_recs)

userPool = []
userPool.append(u1)
userPool.append(jenny)
userPool.append(u3)
userPool.append(u4)
userPool.append(u5)
userPool.append(u6)
userPool.append(u7)
userPool.append(u8)
userPool.append(u9)
userPool.append(u10)

currentTargetNum = 2
currentTarget = jenny
currentTargetSong = t2
recsNum = 10

collabRecs = collabFilter(songPool, userPool, (currentTargetNum - 1), recsNum)

#printTargSong(songPool[currentTargetSong])
#print("Target User's Already Liked Songs")
#for song in currentTarget:
    #printSong(song)

print("Finding new songs for Jenny")

print("Collaboration Recommended Songs")
for song in collabRecs:
    printTitle(song)


