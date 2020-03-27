#!/usr/bin/env python3

import numpy as np
from numpy import inf

class ACO:
    def __init__(self, dist, cod, max_cod, max_dist, n_pick_order, iteration=100, n_ants=10):
        self.dist_matrix = dist
        self.cod_matrix = cod
        self.max_cod = max_cod
        self.max_dist = max_dist

        self.n_pick_order = n_pick_order # Pick n_pick_order order to create route
        self.n_pick_point = n_pick_order*2

        self.n_points = self.dist_matrix.shape[0]
        self.n_orders = int(self.n_points/2) # total orders of input data     
       
        self.iteration = iteration
        self.n_ants = n_ants

        ### intialization part
        self.e = .5  # evaporation rate
        self.alpha = 1  # pheromone factor
        self.beta = 2  # visibility factor

        ### calculating the visibility of the next city visibility(i,j)=1/dist_matrix(i,j)
        self.visibility = 1/self.dist_matrix
        self.visibility[self.visibility == inf] = 0
        self.visibility_temp = self.visibility

        ### intializing the rute of the ants with size rute(n_ants,n_pick_point)
        self.initial_rute = np.zeros((self.n_ants, self.n_pick_point))
        self.cod_lst = np.zeros((self.n_ants, 1))
        self.dist_lst = np.zeros((self.n_ants, 1))
        self.pheromone = .1*np.ones((self.n_points, self.n_points))  #intializing pheromone present at the paths to the cities
        self.index_dict = np.array([dict() for _ in range(self.n_ants)])

        # Result
        self.best_route_lst = []
        self.min_cost_lst = []

    def get_real_index(self, index, ant):
        """
        :param number index: index of point in sub_visibility matrix.
        :param number ant: index of ant.
        :return: index of point in dist_matrix matrix.
        """
        real_index = self.index_dict[int(ant)].get(index)
        if real_index:
            return int(real_index)
        return int(index)

    def conver_pickup_to_dropoff(self, index_pickup):
        """
        :param number index_pickup: index of pickup point in matrix.
        :return: index of dropoff point in dist_matrix matrix.
        """    
        return index_pickup + self.n_orders

    def update_infomation(self, sub_visibility, sub_pheromone, route, ant):
        index_lst = [self.get_real_index(p - 1, ant) for p in list(route)] # convert point list to index list of matrix
        pickup_index_lst = list(filter(lambda x: x < self.n_orders, index_lst)) # finding index list of pickup points in index_lst
        dropoff_index_lst = list(filter(lambda x: x >= self.n_orders, index_lst)) # finding index list of dropoff points in index_lst

        dropoff_index = self.conver_pickup_to_dropoff(pickup_index_lst[-1])  # finding 'index of pickup point' based on last pickup point   
        if dropoff_index not in dropoff_index_lst:
            point_list = list(range(0, self.n_orders)) + list([self.conver_pickup_to_dropoff(i) for i in pickup_index_lst])
            sub_visibility = self.visibility[point_list, :][:, point_list]
            sub_pheromone = self.pheromone[point_list, :][:, point_list]
            self.index_dict[ant].update({len(point_list)-1: dropoff_index})

        if len(pickup_index_lst) == self.n_pick_order:
            index_route_lst = [int(p-1) for p in list(route)]
            lst = list(range(self.n_orders)) + list(filter(lambda x: x >= self.n_orders, list(index_route_lst)))
            for i in lst:
                sub_visibility[:, i] = 0
        else:
            for i in route:
                sub_visibility[:, int(i)-1] = 0        

        return sub_visibility, sub_pheromone

    def pick_initial_point(self):
        for x in range(self.n_orders):
            col = self.visibility_temp[:,x]
            is_diff_zero = list(filter(lambda x: x != 0, list(col)))
            if len(is_diff_zero) !=0:
                return x + 1
        return -1

    def check_condition(self, ant, route):
        last_index = int(route[-1]) - 1
        if last_index < self.n_orders: 
            self.cod_lst[ant] += self.cod_matrix[last_index]
        if len(route) > 2:
            index_lst = [self.get_real_index(p - 1, ant) for p in list(route[-2:])]
            self.dist_lst[ant] += self.dist_matrix[index_lst[0], index_lst[1]]

        if self.cod_lst[ant] > self.max_cod or self.dist_lst[ant] > self.max_dist:
            return True
        return False

    def update_route(self, ant, route):
        index_lst = [self.get_real_index(p - 1, ant) for p in list(route)]
        new_route = []
        for i in range(len(index_lst)):
            if index_lst[i] < self.n_orders and self.conver_pickup_to_dropoff(index_lst[i]) in index_lst:
                new_route.append(route[i])
            elif index_lst[i] >= self.n_orders:
                new_route.append(route[i])    
        new_route += [0 for _ in range(self.n_pick_point - len(new_route))]
        return new_route
               
    def run(self):
        n_approve_point = 0
        n_route = 0
        while(True):
            ### Pick initial point for route
            initial_point = self.pick_initial_point()
            if initial_point == -1:
                break
            for i in range(self.n_ants):
                self.initial_rute[i,0] = initial_point
            pheromone = np.array(self.pheromone)
            visibility = np.array(self.visibility)
            for ite in range(self.iteration):
                rute = self.initial_rute.copy()
                self.cod_lst = np.zeros((self.n_ants, 1))
                self.dist_lst = np.zeros((self.n_ants, 1))
                self.index_dict = np.array([dict() for _ in range(self.n_ants)])
                for i in range(self.n_ants):
                    sub_visibility = np.array(visibility[:self.n_orders,:self.n_orders])  # creating a copy for 'pickup point' of visibility
                    sub_pheromone = np.array(pheromone[:self.n_orders,:self.n_orders]) # creating a copy for 'pickup point' of pheromone
                    check_last_point = True
                    for j in range(self.n_pick_point-2): # expect initial point and last point
                        combine_feature = np.zeros(self.n_points) # intializing combine_feature array to zero            
                        cum_prob = np.zeros(self.n_points) # intializing cummulative probability array to zeros
                        sub_visibility, sub_pheromone = self.update_infomation(sub_visibility, sub_pheromone, rute[i,:j+1], i)
                        if self.check_condition(i, rute[i,:j+1]):
                            check_last_point = False
                            rute[i] = self.update_route(i, rute[i,:j+1])
                            break

                        cur_loc = int(rute[i, j]-1)  # current city of the ant
                        p_feature = np.power(sub_pheromone[cur_loc, :], self.beta) # calculating pheromone feature
                        p_feature = p_feature[:, np.newaxis] # adding axis to make a size[5,1]            
                        v_feature = np.power(sub_visibility[cur_loc, :], self.alpha) # calculating visibility feature
                        v_feature = v_feature[:, np.newaxis] # adding axis to make a size[5,1]

                        combine_feature = np.multiply(p_feature, v_feature) # calculating the combine feature
                        total = np.sum(combine_feature)  # sum of all the feature
                    
                        probs = combine_feature/total # finding probability of element probs(i) = comine_feature(i)/total
                        cum_prob = np.cumsum(probs)  # calculating cummulative sum

                        r = np.random.random_sample()  # random no in [0,1)
                        point_lst = np.nonzero(cum_prob > r)[0]
                        if len(point_lst)!=0:
                            city = point_lst[0]+1 # finding the next city having probability higher then random(r)
                        else:
                            check_last_point = False
                            rute[i] = self.update_route(i, rute[i,:j+1])
                            break  
                        rute[i, j+1] = city  # adding city to route
                        if (n_approve_point + j + 2) == self.n_orders*2:
                            check_last_point = False
                            break      
                    
                    ### Define last point of best route
                    index_lst = [self.get_real_index(p - 1, i) for p in list(rute[i,:-1])]
                    rute[i] = [x + 1 for x in index_lst] + [0]
                    if check_last_point:                
                        pickup_index_lst = list(filter(lambda x: x < self.n_orders, index_lst))
                        all_dropoff_index_set = set([self.conver_pickup_to_dropoff(i) for i in pickup_index_lst])
                        dropoff_index_set = set(filter(lambda x: x >= self.n_orders, index_lst))
                        last_point = list(all_dropoff_index_set - dropoff_index_set)[0] + 1# finding the last untraversed city to route
                        rute[i, -1] = last_point

                ### Calcualting total distance for each route
                rute_opt = np.array(rute)  # intializing optimal route
                dist_cost = np.zeros((self.n_ants, 1)) # intializing total_distance_of_tour with zero
                for i in range(self.n_ants):
                    s = 0
                    for j in range(self.n_pick_point-1):
                        if rute_opt[i, j+1] == 0:
                            break
                        index_x = self.get_real_index(rute_opt[i, j]-1, i)
                        index_y = self.get_real_index(rute_opt[i, j+1]-1, i)
                        s = s + self.dist_matrix[index_x, index_y] # calcualting total tour distance
                    dist_cost[i] = s # storing distance of tour for 'i'th ant at location 'i'

                ### Updating the pheromone
                pheromone = (1-self.e)*pheromone  # evaporation of pheromone with (1-e)
                for i in range(self.n_ants):
                    for j in range(self.n_pick_point-1):
                        if rute_opt[i, j+1] == 0:
                            break
                        dt = 1/dist_cost[i]
                        index_x = self.get_real_index(rute_opt[i, j]-1, i)
                        index_y = self.get_real_index(rute_opt[i, j+1]-1, i)
                        pheromone[index_x, index_y] = pheromone[index_x, index_y] + dt
                        # updating the pheromone with delta_distance
                        # delta_distance will be more with min_dist i.e adding more weight to that route  peromne

                ### finding location of minimum of dist_cost
                dist_min = 100000000000000000
                dist_min_loc = np.argmin(dist_cost)
                if dist_cost[dist_min_loc] == 0.0:
                    for i in range(len(dist_cost)):
                        if dist_cost[i] == 0.0:
                            continue
                        if dist_cost[i] < dist_min:
                            dist_min = dist_cost[i]
                            dist_min_loc = i
                        
                dist_min_cost = dist_cost[dist_min_loc]  # finging min of dist_cost
                best_route = rute[dist_min_loc, :]  # intializing current traversed as best route
                best_route = list(filter(lambda x: x!=0, best_route))

            ### Updating the visibility
            for r in best_route:
                self.visibility_temp[:, int(r) - 1] = 0

            ### Updating picked point number 
            n_approve_point += len(best_route)
            self.best_route_lst.append(best_route)
            self.min_cost_lst.append(float(dist_min_cost[0]))

            # n_route += 1
            # print('Route %d' %(n_route))
            # print('\t +) Best path (%d points):' % len(best_route), best_route)
            # print('\t +) Cost of the best path: %.2f' % dist_min_cost[0])
            # print('-------------------------------------')            
            if n_approve_point == self.n_points:
                break
        return self.best_route_lst, self.min_cost_lst

    def print_result(self):
        if self.best_route_lst:
            total_cost = 0.0
            for i in range(len(self.best_route_lst)):
                best_route = self.best_route_lst[i]
                min_cost = self.min_cost_lst[i]
                total_cost += min_cost
                print('Route %d' %(i+1))
                print('\t +) Best path (%d points):' % len(best_route), best_route)
                print('\t +) Cost of the best path: %.2f' % min_cost)
                print('-------------------------------------')
            print('*** Total cost: %.2f' % total_cost)


if __name__ == '__main__':
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
    max_cod = 15000
    max_dist = 10000
    n_pick_order = 4

    aco_algorithm = ACO(dist_matrix, cod_matrix, max_cod, max_dist, n_pick_order)
    best_route_lst, min_cost_lst = aco_algorithm.run()
    aco_algorithm.print_result()