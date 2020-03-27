#!/usr/bin/env python3

import algorithms
from algorithms.assigning_system_with_simplex import SimplexSolver
from algorithms.combining_system_with_aco import ACO
from algorithms.utils import *
import numpy as np

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
    b = [-1 for _ in range(n_drivers*n_routes)] + [min(n_routes, n_drivers)]
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
    orders = read_orders_from_file('./data/orders_data.csv')
    drivers = read_drivers_from_file('./data/drivers_data.csv')

    points_lst, cod_lst = convert_orders_to_points(orders)
    dist_matrix = calculate_dist_matrix(points_lst)
    cod_matrix = np.array(cod_lst)
    max_cod = 1400000
    max_dist = 1500000
    n_pick_order = 20

    aco_algorithm = ACO(dist_matrix, cod_matrix, max_cod, max_dist, n_pick_order)
    best_route_lst, min_cost_lst = aco_algorithm.run()
    aco_algorithm.print_result()

    A, b, c, p = convert_output1_to_input2(orders, drivers, best_route_lst)
    # SimplexSolver().run_simplex(A,b,c,prob=p,enable_msg=False,latex=True)
