#!/usr/bin/env python3

import math
import numpy as np

def distance(point1: dict, point2: dict): 
    lat1 = point1['lat'] 
    lat2 = point2['lat'] 
    long1 = point1['long'] 
    long2 = point2['long'] 
    try: 
        cost1 = math.cos(math.radians(90-lat1))
        cost2 = math.cos(math.radians(90-lat2))
        sin1 = math.sin(math.radians(90-lat1))
        sin2 = math.sin(math.radians(90-lat2)) 
        cos_delta = math.cos(math.radians(long1-long2)) 
        temp = cost1 * cost2 + sin1 * sin2 * cos_delta 
        if temp > 1: 
            temp = 1
        result = math.acos(temp)*6371 
        return result 
    except Exception as e: 
        print(e) 

def read_orders_from_file(file_path):
    orders = []
    f = open(file_path, "r")    
    index_line = 1
    for line in f:
        if index_line == 1:
            index_line += 1
            continue
        order_data = []
        line_data = line.split(",")
        order_data.append(line_data[0]) # add id of order
        order_data.append((float(line_data[1]), float(line_data[2])))
        order_data.append((float(line_data[3]), float(line_data[4])))
        order_data.append(float(line_data[5]))
        orders.append(order_data)        
        index_line += 1
    f.close()
    return orders

def read_drivers_from_file(file_path):
    drivers = []
    f = open(file_path, "r")    
    index_line = 1
    for line in f:
        if index_line == 1:
            index_line += 1
            continue
        line_data = line.split(",")

        driver_data = []
        # driver_data.append(line_data[0]) # add id of order
        driver_data.append((float(line_data[1]), float(line_data[2])))
        index_line += 1

        drivers.append(driver_data)        
    f.close()
    return drivers

def convert_orders_to_points(orders):
    pickup_point = []
    dropoff_point = []
    cod_lst = []
    for order in orders:
        pickup_point.append(order[1])
        dropoff_point.append(order[2])
        cod_lst.append(order[3])
    return pickup_point + dropoff_point, cod_lst
    
def calculate_dist_matrix(points):
    n_points = len(points)
    dist_matrix = np.zeros((n_points, n_points))
    for x in range(n_points):
        for y in range(x+1, n_points):
            if x == y:
                dist = 0.0
            else:
                dist = distance({'lat': points[x][0], 'long': points[x][1]},
                                {'lat': points[y][0], 'long': points[y][1]})
            dist_matrix[x][y] = dist
            dist_matrix[y][x] = dist
    return dist_matrix

if __name__ == "__main__":
    orders = read_orders_from_file('./data/orders_data.csv')
    drivers = read_drivers_from_file('./data/drivers_data.csv')
    points_lst, cod_lst = convert_orders_to_points(orders)
    dist_matrix = calculate_dist_matrix(points_lst)
    print(len(points_lst))
    print("------------------")
    print(len(cod_lst))