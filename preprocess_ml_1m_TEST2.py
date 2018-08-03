import csv
import numpy as np
from random import shuffle
import random
import collections

line_id = {}
id_line = {}
movie_line = {}
id_movie = {}
id_genre = {}
u_mid_pos = {}
u_mid_pos_test = {}
u_mid_neg = {}
count = 0
mid_genre = {}

'''
Read movie id
'''
with open("./ml-1m/movies.dat", encoding='latin-1') as f:
    for line in f:
        line = line.strip().split('::')

        if line[1] in movie_line:
            line[1] = line[1] + "--2"
            print(line)

        movie_line.update({line[1]: count})
        line_id.update({count: line[0]})
        id_line.update({line[0]: count})
        count = count + 1

        ##genre
        mid_genre.update({line[0]: line[2].split("|")})

'''
split test set and training set
'''
test_size = 0.2
with open("./ml-1m/ratings.dat",encoding='latin-1') as f:
    count = 0
    for line in f:
        line = line.strip().split("::")
        mid_rating = set()
        mid_rating = (id_line[line[1]], line[2])
        if (float(line[2]) < 3):
            u_mid_neg.setdefault(int(line[0]), []).append(mid_rating)
        elif (float(line[2]) > 3):
            count += 1
            if(random.random() <= test_size):
                u_mid_pos_test.setdefault(int(line[0]), []).append(mid_rating)
            else:
                u_mid_pos.setdefault(int(line[0]), []).append(mid_rating)


'''
delete user with few movie ratings
'''
filter_threshold = 20
count = 0

for key, value in u_mid_pos.copy().items():
    if (len(value) < filter_threshold):
        count += 1
        del(u_mid_pos[key])
        if key in u_mid_pos_test:
            del(u_mid_pos_test[key])
    else:
        shuffle(u_mid_pos[key])
print("number of delete", count)

'''
delete user in test set but not in training set
'''
for key, value in u_mid_pos_test.copy().items():
    if key not in u_mid_pos:
        del (u_mid_pos_test[key])


'''
read other features
'''
user_gender = {}  ## 0 is M, 1 is F
user_age = {}  ##  0 - 1, 1 - 18, 2 - 25.....
user_occupation = {} ## 0-0
##### add user related information
with open("./ml-1m/users.dat",encoding='latin-1') as f:
    for line in f:
        line = line.strip().split("::")
        if(line[1] == "M"):
            user_gender.update({int(line[0]): 0})
        else:
            user_gender.update({int(line[0]): 1})

        ### update age
        user_age.update({int(line[0]): int(line[2])/56})

        ###update ocupation
        user_occupation.update({int(line[0]): int(line[3])})

user_genre = {}
for key, value in u_mid_pos.items():
    genre_count = np.zeros(18)
    for index in value:
        id = line_id[index[0]]
        genres = mid_genre[id]
        for genre in genres:
            if (genre == "Action"):
                genre_count[0] += 1
            elif (genre == "Adventure"):
                genre_count[1] += 1
            elif (genre == "Animation"):
                genre_count[2] += 1
            elif (genre == "Children's"):
                genre_count[3] += 1
            elif (genre == "Comedy"):
                genre_count[4] += 1
            elif (genre == "Crime"):
                genre_count[5] += 1
            elif (genre == "Documentary"):
                genre_count[6] += 1
            elif (genre == "Drama"):
                genre_count[7] += 1
            elif (genre == "Fantasy"):
                genre_count[8] += 1
            elif (genre == "Film-Noir"):
                genre_count[9] += 1
            elif (genre == "Horror"):
                genre_count[10] += 1
            elif (genre == "Musical"):
                genre_count[11] += 1
            elif (genre == "Mystery"):
                genre_count[12] += 1
            elif (genre == "Romance"):
                genre_count[13] += 1
            elif (genre == "Sci-Fi"):
                genre_count[14] += 1
            elif (genre == "Thriller"):
                genre_count[15] += 1
            elif (genre == "War"):
                genre_count[16] += 1
            elif (genre == "Western"):
                genre_count[17] += 1
    genre_count = np.divide(genre_count, np.sum(genre_count))
    user_genre.update({key: genre_count})

print("u_mid_pos", len(u_mid_pos))
print("end")


