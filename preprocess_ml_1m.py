import csv
import numpy as np
from random import shuffle

line_id = {}
id_line = {}
movie_line = {}
id_movie = {}
id_genre = {}
u_mid_pos = {}
u_mid_pos_test = {}
u_mid_neg = {}
u_mid_neg_test = {}

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

print(count)
print(len(id_line))



with open("./ml-1m/ratings.dat",encoding='latin-1') as f:
    for line in f:
        line = line.strip().split("::")
        mid_rating = set()
        mid_rating = (id_line[line[1]], line[2])
        if (id_line[line[1]] > 3882):
            print("!!!!!")
        if (float(line[2]) > 3):
            u_mid_pos.setdefault(int(line[0]), []).append(mid_rating)
        elif (float(line[2]) < 3):
            u_mid_neg.setdefault(int(line[0]), []).append(mid_rating)



##filter value < 11
count = 0
for key, value in u_mid_pos.copy().items():
    if (len(value) < 8):
        print("delete")
        del(u_mid_pos[key])
        if key in u_mid_neg:
            del(u_mid_neg[key])
    else:
        shuffle(u_mid_pos[key])

print(len( u_mid_neg))

## split test and training
K = 200
test_count = 0
for key, value in u_mid_pos.copy().items():
    if (test_count <= K):
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
