from collections import deque
from city        import distance, GeoCity, Euc_2D, GeoCoord
from tspparse    import read_tsp_file
from numpy       import array
from pprint      import pprint

def calc_distance(tsp, city1_index, city2_index):
    cities = tsp["CITIES"]
    return distance(cities[city1_index], cities[city2_index])

def tour_distance(tsp, tour):
    total = 0
    for i in tour[:-2]:
        for j in tour[1:]:
            total += calc_distance(tsp, i, j)
    return total

def nearest_neighbor(tsp, start):
    tour = [start]
    remaining = range(len(tsp["CITIES"]))
    del remaining[start]
    while remaining:
        curr = tour[-1]
        dists = [(lambda p, q: calc_distance(tsp, p, q))(p, curr) for p in remaining] 
        indexmin = dists.index(min(dists))  
        tour.append(remaining.pop(indexmin))
    tour.append(start)  # loop back to starting point
    return tour 

def swap_2opt(tsp, tour):
    pass    

def calc_serial_2opt_tour(tsp):
    tour = nearest_neighbor(tsp, 11)
    print tour
    total_dist = tour_distance(tsp, tour)
    tour = swap_2opt(tsp, tour)
    #pprint(tsp)         
    cities = tsp["CITIES"]
    f = lambda k: 'x: {x} y: {y}'.format(x=k.coord_tuple()[0], y=k.coord_tuple()[1])
    strs = map(f, cities)
    #pprint(strs)
    return total_dist

def calc_parallel_2opt_tour(tsp):
    pprint(tsp)         
    cities = tsp["CITIES"]
    f = lambda k: 'x: {x} y: {y}'.format(x=k.coord_tuple()[0], y=k.coord_tuple()[1])
    strs = map(f, cities)
    pprint(strs)

