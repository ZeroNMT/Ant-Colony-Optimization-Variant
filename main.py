#!/usr/bin/env python3

import algorithms
from algorithms.assigning_system_with_simplex import SimplexSolver
from algorithms.combining_system_with_aco import ACO
from algorithms.utils import *
import numpy as np
import time

def convert_output1_to_input2(orders, drivers, best_route_lst):
    n_routes = len(best_route_lst)
    n_drivers = len(drivers)
    A = []
    # calculate each row
    for d in range(n_drivers):
        line = [0 for _ in range(n_routes * n_drivers)]
        start = d*n_routes
        end = start + n_routes
        for x in range(start,end):
            line[x] = -1
        A.append(line)

    # calculate each column
    for d in range(n_routes):
        line = [0 for _ in range(n_routes * n_drivers)]
        for x in range(d,n_routes*n_drivers,n_routes):
            line[x] = -1
        A.append(line)
    A.append([1 for _ in range(n_routes*n_drivers)])
    b = [-1 for _ in range(n_drivers+n_routes)] + [min(n_routes, n_drivers)]
    p = "min"

    start_point_lst = []
    for route in best_route_lst:
        if route:  
            start_point_lst.append(route[0])
    c = [distance({'lat': drivers[x][0][0],'long': drivers[x][0][1]},
                  {'lat': orders[int(y) - 1][1][0],'long': orders[int(y) - 1][1][1]})
                  for x in range(n_drivers) for y in start_point_lst]
    return A,b,c,p

if __name__ == "__main__":
    start = time.time()
    orders = read_orders_from_file('./data/orders_data.csv')
    drivers = read_drivers_from_file('./data/drivers_data.csv')
    points_lst, cod_lst = convert_orders_to_points(orders)
    dist_matrix = calculate_dist_matrix(points_lst)
    cod_matrix = np.array(cod_lst)
    step1 = time.time()
    ### Please enter input value:
    max_cod = 1400000
    max_dist = 1500000
    n_pick_order = 10

    ### Part 1
    aco_algorithm = ACO(dist_matrix, cod_matrix, max_cod, max_dist, n_pick_order, iteration=100, n_ants=10)
    best_route_lst, min_cost_lst = aco_algorithm.run()
    step2 = time.time()

    ### Part 2
    A, b, c, p = convert_output1_to_input2(orders, drivers, best_route_lst)
    obj = SimplexSolver()
    obj.run_simplex(A,b,c,prob=p,enable_msg=False,latex=False)
    result = obj._get_result(len(drivers), len(best_route_lst))
    step3 = time.time()

    ### Print result
    total_cost = 0.0
    for i in range(len(best_route_lst)):
        best_route = best_route_lst[i]
        min_cost = min_cost_lst[i]
        print('Route %d' %(i+1))
        print('\t +) Best path (%d points):' % len(best_route), best_route)
        print('\t +) Cost of the best path: %.2f' % min_cost)
        driver = result.get(i+1)
        print('\t +) Driver: ', driver if driver else 'NO')
        if driver:
            distance_driver = c[(driver-1)*len(best_route_lst) + i]
            min_cost += distance_driver
            print('\t +) Distance from driver to starting point: %.2f' % distance_driver)
            print('\t +) Traveled distance by the driver: %.2f' % min_cost)
        print('-------------------------------------')
        total_cost += min_cost

    print('*** Total cost: %.2f' % total_cost)
    read_file_time = step1 -start
    part1_time = step2 - step1
    part2_time = step3 - step2
    print('*** Read files: %.2f seconds = %.6f minutes' % (read_file_time, read_file_time/60.0) )
    print('*** Part 1: %.2f seconds = %.6f minutes' % (part1_time, part1_time/60.0) )
    print('*** Part 2: %.2f seconds = %.6f minutes' % (part2_time, part2_time/60.0) )
