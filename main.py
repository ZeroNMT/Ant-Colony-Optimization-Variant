import numpy as np
from numpy import inf

### given values for the problems
dist_matrix = np.array([
[0, 11, 13, 15, 78, 34, 43, 54, 76, 26, 84, 25, 13, 64, 21, 85, 24, 83, 12, 12],
[11, 0, 11, 23, 24, 17, 13, 32, 85, 32, 84, 18, 31, 37, 56, 32, 28, 24, 43, 53],
[13, 11, 0, 36, 75, 23, 42, 5, 45, 46, 75, 11, 76, 44, 22, 69, 53, 63, 76, 64],
[15, 23, 36, 0, 31, 75, 16, 32, 12, 22, 45, 86, 21, 21, 74, 42, 21, 32, 27, 86],
[78, 24, 75, 31, 0, 53, 67, 87, 31, 35, 75, 23, 21, 21, 7, 53, 87, 14, 23, 34],
[34, 17, 23, 75, 53, 0, 13, 99, 72, 18, 12, 32, 74, 42, 12, 24, 52, 64, 54, 76],
[43, 13, 42, 16, 67, 13, 0, 80, 25, 41, 42, 67, 90, 24, 11, 13, 32, 32, 87, 43],
[54, 32, 5, 32, 87, 99, 80, 0, 55, 16, 11, 77, 54, 75, 32, 11, 16, 86, 25, 57],
[76, 85, 45, 12, 31, 72, 25, 55, 0, 79, 98, 43, 66, 79, 57, 36, 11, 34, 24, 43],
[26, 32, 46, 22, 35, 18, 41, 16, 79, 0, 57, 42, 87, 97, 90, 32, 12, 21, 43, 69],
[84, 84, 75, 45, 75, 12, 42, 11, 98, 57, 0, 85, 45, 43, 21, 74, 23, 57, 86, 43],
[25, 18, 11, 86, 23, 32, 67, 77, 43, 42, 85, 0, 55, 12, 42, 86, 53, 42, 31, 53],
[13, 31, 76, 21, 21, 74, 90, 54, 66, 87, 45, 55, 0, 42, 7, 96, 67, 54, 41, 21],
[64, 37, 44, 21, 21, 42, 24, 75, 79, 97, 43, 12, 42, 0, 24, 31, 53, 32, 56, 35],
[21, 56, 22, 74, 7, 12, 11, 32, 57, 90, 21, 42, 7, 24, 0, 1, 75, 97, 23, 87],
[85, 32, 69, 42, 53, 24, 13, 11, 36, 32, 74, 86, 96, 31, 1, 0, 87, 32, 94, 43],
[24, 28, 53, 21, 87, 52, 32, 16, 11, 12, 23, 53, 67, 53, 75, 87, 0, 34, 89, 32],
[83, 24, 63, 32, 14, 64, 32, 86, 34, 21, 57, 42, 54, 32, 97, 32, 34, 0, 43, 18],
[12, 43, 76, 27, 23, 54, 87, 25, 24, 43, 86, 31, 41, 56, 23, 94, 89, 43, 0, 53],
[12, 53, 64, 86, 34, 76, 43, 57, 43, 69, 43, 53, 21, 35, 87, 43, 32, 18, 53, 0]
])
cod_matrix = np.array([
    540.0, 340.0, 150.0, 230.0, 450.0, 340.0,
    150.0, 340.0, 780.0, 140.0, 440.0])
fee_matrix = np.array([
    [0, 11, 13, 15, 78, 34, 43, 54, 76, 26, 84, 25, 13, 64, 21, 85, 24, 83, 12, 12],
[11, 0, 11, 23, 24, 17, 13, 32, 85, 32, 84, 18, 31, 37, 56, 32, 28, 24, 43, 53],
[13, 11, 0, 36, 75, 23, 42, 5, 45, 46, 75, 11, 76, 44, 22, 69, 53, 63, 76, 64],
[15, 23, 36, 0, 31, 75, 16, 32, 12, 22, 45, 86, 21, 21, 74, 42, 21, 32, 27, 86],
[78, 24, 75, 31, 0, 53, 67, 87, 31, 35, 75, 23, 21, 21, 7, 53, 87, 14, 23, 34],
[34, 17, 23, 75, 53, 0, 13, 99, 72, 18, 12, 32, 74, 42, 12, 24, 52, 64, 54, 76],
[43, 13, 42, 16, 67, 13, 0, 80, 25, 41, 42, 67, 90, 24, 11, 13, 32, 32, 87, 43],
[54, 32, 5, 32, 87, 99, 80, 0, 55, 16, 11, 77, 54, 75, 32, 11, 16, 86, 25, 57],
[76, 85, 45, 12, 31, 72, 25, 55, 0, 79, 98, 43, 66, 79, 57, 36, 11, 34, 24, 43],
[26, 32, 46, 22, 35, 18, 41, 16, 79, 0, 57, 42, 87, 97, 90, 32, 12, 21, 43, 69],
[84, 84, 75, 45, 75, 12, 42, 11, 98, 57, 0, 85, 45, 43, 21, 74, 23, 57, 86, 43],
[25, 18, 11, 86, 23, 32, 67, 77, 43, 42, 85, 0, 55, 12, 42, 86, 53, 42, 31, 53],
[13, 31, 76, 21, 21, 74, 90, 54, 66, 87, 45, 55, 0, 42, 7, 96, 67, 54, 41, 21],
[64, 37, 44, 21, 21, 42, 24, 75, 79, 97, 43, 12, 42, 0, 24, 31, 53, 32, 56, 35],
[21, 56, 22, 74, 7, 12, 11, 32, 57, 90, 21, 42, 7, 24, 0, 1, 75, 97, 23, 87],
[85, 32, 69, 42, 53, 24, 13, 11, 36, 32, 74, 86, 96, 31, 1, 0, 87, 32, 94, 43],
[24, 28, 53, 21, 87, 52, 32, 16, 11, 12, 23, 53, 67, 53, 75, 87, 0, 34, 89, 32],
[83, 24, 63, 32, 14, 64, 32, 86, 34, 21, 57, 42, 54, 32, 97, 32, 34, 0, 43, 18],
[12, 43, 76, 27, 23, 54, 87, 25, 24, 43, 86, 31, 41, 56, 23, 94, 89, 43, 0, 53],
[12, 53, 64, 86, 34, 76, 43, 57, 43, 69, 43, 53, 21, 35, 87, 43, 32, 18, 53, 0]
])
iteration = 100
n_ants = 10
n_pick_order = 8 # Pick n_pick_order order to create route
n_orders = 10 # total orders of input data


### intialization part
n_pick_point = n_pick_order*2
m = n_ants
n = n_orders*2
e = .5  # evaporation rate
alpha = 1  # pheromone factor
beta = 2  # visibility factor


### calculating the visibility of the next city visibility(i,j)=1/dist_matrix(i,j)
visibility = 1/dist_matrix
visibility[visibility == inf] = 0

# intializing the rute of the ants with size rute(n_ants,n_pick_point+1)
# note adding 1 because we want to come back to the source city
rute = np.ones((m, n_pick_point + 1))

pheromone = .1*np.ones((n, n))  #intializing pheromone present at the paths to the cities

index_dict = np.array([dict() for _ in range(m)])
def get_real_index(index, ant):
    """
       :param number index: index of point in sub_visibility matrix.
       :param number ant: index of ant.
       :return: index of point in dist_matrix matrix.
    """
    real_index = index_dict[int(ant)].get(index)
    if real_index:
        return int(real_index)
    return int(index)

def conver_pickup_to_dropoff(index_pickup):
    """
       :param number index_pickup: index of pickup point in matrix.
       :return: index of dropoff point in dist_matrix matrix.
    """    
    return index_pickup + n_orders

def update_infomation(sub_visibility, sub_pheromone, route, ant):
    index_lst = [get_real_index(p - 1, ant) for p in list(route)] # convert point list to index list of matrix
    pickup_index_lst = list(filter(lambda x: x < n_orders, index_lst)) # finding index list of pickup points in index_lst
    dropoff_index_lst = list(filter(lambda x: x >= n_orders, index_lst)) # finding index list of dropoff points in index_lst

    dropoff_index = conver_pickup_to_dropoff(pickup_index_lst[-1])  # finding 'index of pickup point' based on last pickup point   
    if dropoff_index not in dropoff_index_lst:
        point_list = list(range(0,n_orders)) + list([conver_pickup_to_dropoff(i) for i in pickup_index_lst])
        sub_visibility = visibility[point_list, :][:, point_list]
        sub_pheromone = pheromone[point_list, :][:, point_list]
        index_dict[ant].update({len(point_list)-1: dropoff_index})

    if len(pickup_index_lst) == n_pick_order:
        index_route_lst = [int(p-1) for p in list(route)]
        lst = list(range(n_orders)) + list(filter(lambda x: x >= n_orders, list(index_route_lst)))
        for i in lst:
            sub_visibility[:, i] = 0
    else:
        for i in route:
            sub_visibility[:, int(i)-1] = 0        

    return sub_visibility, sub_pheromone

for ite in range(iteration):
    index_dict = np.array([dict() for _ in range(m)])
    rute = np.ones((m, n_pick_point + 1))
    for i in range(m):
        sub_visibility = np.array(visibility[:n_orders,:n_orders])  # creating a copy for 'pickup point' of visibility
        sub_pheromone = np.array(pheromone[:n_orders,:n_orders]) # creating a copy for 'pickup point' of pheromone
        for j in range(n_pick_point-2): # expect initial point and last point
            combine_feature = np.zeros(n) # intializing combine_feature array to zero            
            cum_prob = np.zeros(n) # intializing cummulative probability array to zeros
            sub_visibility, sub_pheromone = update_infomation(sub_visibility, sub_pheromone, rute[i,:j+1], i)

            cur_loc = int(rute[i, j]-1)  # current city of the ant
            p_feature = np.power(sub_pheromone[cur_loc, :], beta) # calculating pheromone feature
            p_feature = p_feature[:, np.newaxis] # adding axis to make a size[5,1]            
            v_feature = np.power(sub_visibility[cur_loc, :], alpha) # calculating visibility feature
            v_feature = v_feature[:, np.newaxis] # adding axis to make a size[5,1]

            combine_feature = np.multiply(p_feature, v_feature) # calculating the combine feature
            total = np.sum(combine_feature)  # sum of all the feature
           
            probs = combine_feature/total # finding probability of element probs(i) = comine_feature(i)/total
            cum_prob = np.cumsum(probs)  # calculating cummulative sum

            r = np.random.random_sample()  # random no in [0,1)
            city = np.nonzero(cum_prob > r)[0][0]+1 # finding the next city having probability higher then random(r)
            rute[i, j+1] = city  # adding city to route
            
        index_lst = [get_real_index(p - 1, i) for p in list(rute[i,:-2])]
        pickup_index_lst = list(filter(lambda x: x < n_orders, index_lst))
        all_dropoff_index_set = set([conver_pickup_to_dropoff(i) for i in pickup_index_lst])
        dropoff_index_set = set(filter(lambda x: x >= n_orders, index_lst))
        last_point = list(all_dropoff_index_set - dropoff_index_set)[0] + 1# finding the last untraversed city to route
        rute[i] = [x + 1 for x in index_lst] + [last_point, index_lst[0] + 1]

    ### Calcualting total distance for each route
    rute_opt = np.array(rute)  # intializing optimal route
    dist_cost = np.zeros((m, 1)) # intializing total_distance_of_tour with zero
    for i in range(m):
        s = 0
        for j in range(n_pick_point-1):
            index_x = get_real_index(rute_opt[i, j]-1, i)
            index_y = get_real_index(rute_opt[i, j+1]-1, i)
            s = s + dist_matrix[index_x, index_y] # calcualting total tour distance
        dist_cost[i] = s # storing distance of tour for 'i'th ant at location 'i'

    dist_min_loc = np.argmin(dist_cost) # finding location of minimum of dist_cost
    dist_min_cost = dist_cost[dist_min_loc]  # finging min of dist_cost
    best_route = rute[dist_min_loc, :]  # intializing current traversed as best route

    ### Updating the pheromone
    pheromone = (1-e)*pheromone  # evaporation of pheromone with (1-e)
    for i in range(m):
        for j in range(n_pick_point-1):
            dt = 1/dist_cost[i]
            index_x = get_real_index(rute_opt[i, j]-1, i)
            index_y = get_real_index(rute_opt[i, j+1]-1, i)
            pheromone[index_x, index_y] = pheromone[index_x, index_y] + dt
            # updating the pheromone with delta_distance
            # delta_distance will be more with min_dist i.e adding more weight to that route  peromne

print('route of all the ants at the end :')
print(rute_opt)
print()
print('best path :', best_route)
print('cost of the best path', int(
    dist_min_cost[0]) + dist_matrix[int(best_route[-2])-1, 0])
