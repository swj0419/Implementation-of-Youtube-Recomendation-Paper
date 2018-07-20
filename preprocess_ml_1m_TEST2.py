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
test_size = 0.1


count = 0

mid_genre = {}
##Read movie id
with open("./ml-1m/movies.dat", encoding='latin-1') as f:
    for line in f:
        line = line.strip().split('::')

        if line[1] in movie_line:
            line[1] = line[1] + "--2"
            print(line)

        # if line[1] in movie_line:
        #     line[1] = line[1] + "--3"
        #
        #     print(line)
        movie_line.update({line[1]: count})
        line_id.update({count: line[0]})
        id_line.update({line[0]: count})
        count = count + 1
        ##genre
        genre = []
        genre.append(line[2].split("|"))
        mid_genre.update({line[0]: genre})

#####split train and test


user_genre = {}
with open("./ml-1m/ratings.dat",encoding='latin-1') as f:
    count = 0
    for line in f:
        line = line.strip().split("::")
        mid_rating = set()
        mid_rating = (id_line[line[1]], line[2])

        # ## user genre
        # for genre in mid_genre[line[1]]:
        #     if(genre=="Action"):
        #         user_genre.setdefault(int(line[0]), dict).update
        #     elif(genre=="Adventure"):
        #         user_genre[int(line[0])][1] += 1
        #     elif (genre == "Animation"):
        #         user_genre[int(line[0])][2] += 1
        #     elif (genre == "Children's"):
        #         user_genre[int(line[0])][3] += 1
        #     elif (genre == "Comedy"):
        #         user_genre[int(line[0])][4] += 1
        #     elif (genre == "Crime"):
        #         user_genre[int(line[0])][5] += 1
        #     elif (genre == "Documentary"):
        #         user_genre[int(line[0])][6] += 1
        #     elif (genre == "Drama"):
        #         user_genre[int(line[0])][7] += 1
        #     elif (genre == "Fantasy"):
        #         user_genre[int(line[0])][8] += 1
        #     elif (genre == "Film-Noir"):
        #         user_genre[int(line[0])][9] += 1
        #     elif (genre == "Horror"):
        #         user_genre[int(line[0])][10] += 1
        #     elif (genre == "Musical"):
        #         user_genre[int(line[0])][11] += 1
        #     elif (genre == "Mystery"):
        #         user_genre[int(line[0])][12] += 1
        #     elif (genre == "Romance"):
        #         user_genre[int(line[0])][13] += 1
        #     elif (genre == "Sci-Fi"):
        #         user_genre[int(line[0])][14] += 1
        #     elif (genre == "Thriller"):
        #         user_genre[int(line[0])][15] += 1
        #     elif (genre == "War"):
        #         user_genre[int(line[0])][16] += 1
        #     elif (genre == "Western"):
        #         user_genre[int(line[0])][17] += 1
        #



        if (float(line[2]) < 3):
            u_mid_neg.setdefault(int(line[0]), []).append(mid_rating)
        elif (float(line[2]) > 3):
            count += 1
            if(random.random() <= test_size):
                u_mid_pos_test.setdefault(int(line[0]), []).append(mid_rating)
            else:
                u_mid_pos.setdefault(int(line[0]), []).append(mid_rating)


print("count", count)


'''
u_mid_pos has 6038 user, 575281 rates
test size is 0.1 
'''

filter_threshold = 30
##filter value < 8
count = 0

for key, value in u_mid_pos.copy().items():
    if (len(value) < filter_threshold):
        print("delete")
        del(u_mid_pos[key])
        if key in u_mid_pos_test:
            del(u_mid_pos_test[key])
    else:
        count += 1
        shuffle(u_mid_pos[key])


for key, value in u_mid_pos_test.copy().items():
    if key not in u_mid_pos:
        del (u_mid_pos_test[key])


print(len(u_mid_pos))
print(len(u_mid_pos_test))


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
        if(line[2] == "1"):
            user_age.update({int(line[0]): 0})
        elif(line[2] == "18"):
            user_age.update({int(line[0]): 1})
        elif (line[2] == "25"):
            user_age.update({int(line[0]): 2})
        elif (line[2] == "35"):
            user_age.update({int(line[0]): 3})
        elif (line[2] == "45"):
            user_age.update({int(line[0]): 4})
        elif (line[2] == "50"):
            user_age.update({int(line[0]): 5})
        elif (line[2] == "56"):
            user_age.update({int(line[0]): 6})

        ###update ocupation
        user_occupation.update({int(line[0]): int(line[3])})





print("end")


