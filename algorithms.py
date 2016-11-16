from collections import deque
from city        import distance, GeoCity, Euc_2D, GeoCoord
from tspparse    import read_tsp_file
from numpy       import array
from pprint      import pprint
from matplotlib import pyplot as plt
import networkx as nx
import copy
import random

def calc_distance(tsp, city1_index, city2_index):
    cities = tsp["CITIES"]
    return distance(cities[city1_index], cities[city2_index])

def tour_distance(tsp, tour):
    total = 0
    for i,j in zip(tour[:-1], tour[1:]):
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

def swap_2opt(tsp, tour_input):
    tour = copy.deepcopy(tour_input) 
    if tour[0] == tour[-1]:
        del tour[-1]
    MIN_TOTAL_IMPROVE = 100 
    MAX_IMPROVE_LOOP = 50
    MAX_TABU = 1000
    improve = [MIN_TOTAL_IMPROVE+1]
    iteration = 0
    tabu = []
    edge_1 = 0
    edge_2 = 2
    while sum(improve) > MIN_TOTAL_IMPROVE:
        edge_1, edge_2, dist = choose_edge_random(tsp, tour, tabu)
        if edge_1 == -1 or edge_2 == -1:
            print "Invalid edges!"
            break

        if dist > 0:
            if len(tabu) > MAX_TABU:
                del tabu[0]
            city_a = tour[edge_1]
            city_b = tour[edge_2]
            if city_a > city_b:
                x = city_a
                city_a = city_b
                city_b = x
            #print (city_a, city_b)
            #print tour
            tabu.append((city_a, city_b))  
            #print tabu
            #print (tabu[-1], dist)
            tour[edge_1+1:edge_2+1] = tour[edge_1+1:edge_2+1][::-1]
            #print tour
            #print tour

        if len(improve) > MAX_IMPROVE_LOOP:
            del improve[0]
        improve.append(dist)
        #print improve

        iteration += 1
    tour.append(tour[0])
    return tour                 

#choose edges randomly
#maximizing distance reduction
# e1 < e2
def choose_edge_random(tsp, tour, tabu):
    e1, e2 = 0, 2
    e1_best, e2_best, max_dist = e1, e2, 0
    retry = len(tour) * 4 
    while retry > 0: 
        e1 = random.randint(0, len(tour)-1)
        e2 = random.randint(0, len(tour)-1)
        if e1 > e2:
            x = e1
            e1 = e2
            e2 = x
        if (e2 - e1) < 2 or (e1 == 0 and e2 == len(tour)-1): 
            continue
        #print (tour[e1], tour[e2])
        city_a = tour[e1]
        city_b = tour[e2]
        if city_a > city_b:
            x = city_a
            city_a = city_b
            city_b = x
        if (city_a, city_b) in tabu:
            continue 
        retry -= 1
    
        city_1, city_3 = tour[e1], tour[e2]
        city_2 = tour[e1+1] if e1+1 < len(tour) else tour[0]  
        city_4 = tour[e2+1] if e2+1 < len(tour) else tour[0]  

        old_dist = calc_distance(tsp, city_1, city_2) + \
                   calc_distance(tsp, city_3, city_4) 
        new_dist = calc_distance(tsp, city_1, city_3) + \
                   calc_distance(tsp, city_2, city_4) 
        diff_dist = old_dist - new_dist
        #print (e1, e2, diff_dist)
        max_dist = diff_dist if diff_dist > 0 and max_dist == 0 else max_dist

        if diff_dist > 0 and diff_dist > max_dist:
            e1_best, e2_best, max_dist = e1, e2, diff_dist

    return e1_best, e2_best, max_dist

def calc_serial_2opt_tour(tsp):
    tour = nearest_neighbor(tsp, 1)



    iteration=200
    STATE=0
    BESTTOURLEN=10000000
    BESTTOUR= copy.copy(tour)
    MAXTABU=5
    TABULIST= [[] for _ in range(MAXTABU)]
    for i in xrange (MAXTABU):
       TABULIST[i]= copy.copy(tour)
    TABUPTR=1
    for i in xrange (iteration):
       if STATE >=1:
           index=random.randrange(0,len(TABULIST),1)
           print "Random Generated: %d" %index
           tour = swap_2opt(tsp, TABULIST[index])
       else:           
           tour = swap_2opt(tsp, BESTTOUR)
       total_dist = tour_distance(tsp, tour)
       if total_dist < BESTTOURLEN:
          BESTTOURLEN = total_dist
          BESTTOUR =  copy.copy(tour)
          if TABUPTR < MAXTABU:
             TABULIST[TABUPTR]= copy.copy(tour)
             TABUPTR=TABUPTR+1
          else:
             #Replace the first item in tabu list if fully filled
             TABUPTR=0
          if STATE <> 0:
             STATE=STATE-1
       else:
          STATE=STATE+1
          if STATE >= 4:

             break


       
       

    G=nx.Graph()
    pos = {}
    cnt = 0
    for i in tsp["CITIES"]:
        pos.update({ cnt:(i.x, i.y) }) 
        cnt += 1
    G.add_nodes_from(pos)
    edges = []
    for i,j in zip(BESTTOUR[:], BESTTOUR[1:]):
       edges.append((i, j)) 
    G.add_edges_from(edges)
    nx.draw_networkx(G, pos=pos, node_size=100, font_size=6)
    plt.axis('off')
    plt.show() # display

    print BESTTOURLEN
    #pprint(tsp)         
    #cities = tsp["CITIES"]
    #f = lambda k: 'x: {x} y: {y}'.format(x=k.coord_tuple()[0], y=k.coord_tuple()[1])
    #strs = map(f, cities)
    #pprint(strs)
    return BESTTOUR

def calc_parallel_2opt_tour(tsp):
    pprint(tsp)         
    cities = tsp["CITIES"]
    f = lambda k: 'x: {x} y: {y}'.format(x=k.coord_tuple()[0], y=k.coord_tuple()[1])
    strs = map(f, cities)
    pprint(strs)

