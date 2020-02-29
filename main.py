import numpy as np
from numpy import inf

# given values for the problems

dist_matrix = np.array([
    [0, 10, 12, 11, 14, 16, 17, 17, 20, 25],
    [10, 0, 13, 15, 8, 23, 22, 27, 13, 15],
    [12, 13, 0, 9, 14, 14, 8, 14, 16, 13],
    [11, 15, 9, 0, 16, 11, 15, 9, 12, 13],
    [14, 8, 14, 16, 0, 12, 11, 4, 5, 8],
    [12, 10, 12, 11, 14, 0, 11, 15, 9, 10],
    [10, 13, 13, 15, 8, 11, 0, 11, 15, 9],
    [12, 13, 8, 9, 14, 11, 14, 0, 16, 18],
    [11, 15, 9, 23, 16, 11, 15, 9, 0, 3],
    [14, 8, 14, 16, 12, 11, 15, 9, 8, 0]
])
cod_matrix = np.array([
    540.0, 340.0, 150.0, 230.0, 450.0, 340.0,
    150.0, 340.0, 780.0, 140.0, 440.0])
fee_matrix = np.array([
    [0, 10, 12, 11, 14, 16, 17, 17, 20, 25],
    [10, 0, 13, 15, 8, 23, 22, 27, 13, 15],
    [12, 13, 0, 9, 14, 14, 8, 14, 16, 13],
    [11, 15, 9, 0, 16, 11, 15, 9, 12, 13],
    [14, 8, 14, 16, 0, 12, 11, 4, 5, 8],
    [12, 10, 12, 11, 14, 0, 11, 15, 9, 10],
    [10, 13, 13, 15, 8, 11, 0, 11, 15, 9],
    [12, 13, 8, 9, 14, 11, 14, 0, 16, 18],
    [11, 15, 9, 23, 16, 11, 15, 9, 0, 3],
    [14, 8, 14, 16, 12, 11, 15, 9, 8, 0]
])
iteration = 100
n_ants = 10
n_citys = 10

# intialization part

m = n_ants
n = n_citys
e = .5  # evaporation rate
alpha = 1  # pheromone factor
beta = 2  # visibility factor

# calculating the visibility of the next city visibility(i,j)=1/d(i,j)

visibility = 1/dist_matrix
visibility[visibility == inf] = 0

# intializing pheromne present at the paths to the cities

pheromne = .1*np.ones((m, n))

# intializing the rute of the ants with size rute(n_ants,n_citys+1)
# note adding 1 because we want to come back to the source city

rute = np.ones((m, n+1))

for ite in range(iteration):

    # initial starting and ending positon of every ants '1' i.e city '1'
    rute[:, 0] = 1

    for i in range(m):

        temp_visibility = np.array(visibility)  # creating a copy of visibility

        for j in range(n-1):
            # print(rute)

            # intializing combine_feature array to zero
            combine_feature = np.zeros(n)
            # intializing cummulative probability array to zeros
            cum_prob = np.zeros(n)

            cur_loc = int(rute[i, j]-1)  # current city of the ant

            # making visibility of the current city as zero
            temp_visibility[:, cur_loc] = 0

            # calculating pheromne feature
            p_feature = np.power(pheromne[cur_loc, :], beta)
            # calculating visibility feature
            v_feature = np.power(temp_visibility[cur_loc, :], alpha)

            # adding axis to make a size[5,1]
            p_feature = p_feature[:, np.newaxis]
            # adding axis to make a size[5,1]
            v_feature = v_feature[:, np.newaxis]

            # calculating the combine feature
            combine_feature = np.multiply(p_feature, v_feature)

            total = np.sum(combine_feature)  # sum of all the feature

            # finding probability of element probs(i) = comine_feature(i)/total
            probs = combine_feature/total

            cum_prob = np.cumsum(probs)  # calculating cummulative sum
            # print(cum_prob)
            r = np.random.random_sample()  # randon no in [0,1)
            # print(r)
            # finding the next city having probability higher then random(r)
            city = np.nonzero(cum_prob > r)[0][0]+1
            # print(city)

            rute[i, j+1] = city  # adding city to route

        # finding the last untraversed city to route
        left = list(set([i for i in range(1, n+1)])-set(rute[i, :-2]))[0]

        rute[i, -2] = left  # adding untraversed city to route

    rute_opt = np.array(rute)  # intializing optimal route

    # intializing total_distance_of_tour with zero
    dist_cost = np.zeros((m, 1))

    for i in range(m):

        s = 0
        for j in range(n-1):

            # calcualting total tour distance
            s = s + dist_matrix[int(rute_opt[i, j])-1, int(rute_opt[i, j+1])-1]

        # storing distance of tour for 'i'th ant at location 'i'
        dist_cost[i] = s

    # finding location of minimum of dist_cost
    dist_min_loc = np.argmin(dist_cost)
    dist_min_cost = dist_cost[dist_min_loc]  # finging min of dist_cost

    # intializing current traversed as best route
    best_route = rute[dist_min_loc, :]
    pheromne = (1-e)*pheromne  # evaporation of pheromne with (1-e)

    for i in range(m):
        for j in range(n-1):
            dt = 1/dist_cost[i]
            pheromne[int(rute_opt[i, j])-1, int(rute_opt[i, j+1]) -
                     1] = pheromne[int(rute_opt[i, j])-1, int(rute_opt[i, j+1])-1] + dt
            # updating the pheromne with delta_distance
            # delta_distance will be more with min_dist i.e adding more weight to that route  peromne

print('route of all the ants at the end :')
print(rute_opt)
print()
print('best path :', best_route)
print('cost of the best path', int(
    dist_min_cost[0]) + dist_matrix[int(best_route[-2])-1, 0])
