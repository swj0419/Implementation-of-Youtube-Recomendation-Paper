import csv
import numpy as np
from random import shuffle
import random

line_id = {}
id_line = {}
movie_line = {}
id_movie = {}
id_genre = {}
u_mid_pos = {}
u_mid_pos_test = {}
u_mid_neg = {}


firstline = True

count = 0
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
        # genre = []
        # genre.append(line[2].split("|"))
        # id_genre.update({line[0]: genre})

#####split train and test


with open("./ml-1m/ratings.dat",encoding='latin-1') as f:
    count = 0
    for line in f:
        line = line.strip().split("::")
        mid_rating = set()
        mid_rating = (id_line[line[1]], line[2])

        if (float(line[2]) > 3):
            u_mid_pos.setdefault(int(line[0]), []).append(mid_rating)
            count += 1
        elif (float(line[2]) < 3):
            u_mid_neg.setdefault(int(line[0]), []).append(mid_rating)
    print(count)
print(len(u_mid_pos))

'''
u_mid_pos has 6038 user, 575281 rates
test size is 0.1 

'''
test_size = 0.1
filter_threshold = 40
##filter value < 8
count = 0

for key, value in u_mid_pos.copy().items():
    if (len(value) < filter_threshold):
        print("delete")
        del(u_mid_pos[key])
    else:
        count += 1
        shuffle(u_mid_pos[key])




## split test and training
test_count = 0
for key, value in u_mid_pos.copy().items():
    if(test_count<100):
        del(u_mid_pos[key])
        u_mid_pos_test.update({key:value})
    test_count += 1

print(len(u_mid_pos))


print("end")

'''''
    a = dict((k,u_mid_pos[k]) for k in np.arange(1, 5, 1))
    # print(u_mid_pos[k] for k in np.arange(1, 5, 1))

    print(np.arange(11, 17, 1))

    print("end")
'''''
