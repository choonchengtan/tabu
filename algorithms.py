from collections import deque
from city        import distance, GeoCity, Euc_2D, GeoCoord
from tspparse    import read_tsp_file
from numpy       import array
from pprint      import pprint
from multiprocessing import Process, Queue, Pool
from functools   import partial
from itertools   import chain
import copy
import random

def calc_distance(cities,city1_index, city2_index):
    return distance(cities[city1_index], cities[city2_index])
            
def tour_distance_r(cities,tour):
    total = 0
    procs=4
    pool = Pool(processes=procs)
    sizeSegment = len(tour)/procs
    
    jobs = []
    for i in range (0,procs):
        if i==procs:
           jobs.append ((cities, tour[i*sizeSegment:len(tour)]))
        else:
           jobs.append ((cities, tour[i*sizeSegment:(i+1)*sizeSegment]))   
    pool =pool.map(tour_distance,jobs)
    total=sum(pool)
    total += calc_distance(cities,tour[0], tour[len(tour)-1])
    return total

    
def tour_distance(cities, tour):
    total = 0
    for i,j in zip(tour[:-1], tour[1:]):
        total += calc_distance(cities, i, j)
    return total

def just_returnTour(cities):
    tour = range(len(cities))
    return tour
    
def nearest_neighbor(cities,start):
    tour = [start]
    remaining = range(len(cities))
    del remaining[start]
    while remaining:
        curr = tour[-1]
        dists = [(lambda p, q: calc_distance(cities,p, q))(p, curr) for p in remaining] 
        indexmin = dists.index(min(dists))  
        tour.append(remaining.pop(indexmin))
    #tour.append(start)  # loop back to starting point
    return tour 

    
def reverse(city, n):
    nct = len(city)
    nn = (1+ ((n[1]-n[0]) % nct))/2 # half the lenght of the segment to be reversed
    # the segment is reversed in the following way n[0]<->n[1], n[0]+1<->n[1]-1, n[0]+2<->n[1]-2,...
    # Start at the ends of the segment and swap pairs of cities, moving towards the center.
    for j in range(nn):
        k = (n[0]+j) % nct
        l = (n[1]-j) % nct
        (city[k],city[l]) = (city[l],city[k])  # swap
    
def transpt(city, n):
    nct = len(city)
    
    newcity=[]
    # Segment in the range n[0]...n[1]
    for j in range( (n[1]-n[0])%nct + 1):
        newcity.append(city[ (j+n[0])%nct ])
    # is followed by segment n[5]...n[2]
    for j in range( (n[2]-n[5])%nct + 1):
        newcity.append(city[ (j+n[5])%nct ])
    # is followed by segment n[3]...n[4]
    for j in range( (n[4]-n[3])%nct + 1):
        newcity.append(city[ (j+n[3])%nct ])
    return newcity
    
def swap_2opt_SA(cities, tour, dist):
    Route = copy.deepcopy(tour) 
    nct= len(tour)
    n = array([0]*6)
    accepted = 0
    maxAccepted= 10*nct
    maxSteps = 100*nct     # Number of steps at constant temperature    
    Preverse = 0.5         # How often to choose reverse/transpose trial move
    #Tstart = 0.2           # Starting temperature - has to be high enough
    #T = Tstart             # temperature
                
    for i in range(maxSteps): # At each temperature, many Monte Carlo steps
    
            while True: # Will find two random cities sufficiently close by
                # Two cities n[0] and n[1] are choosen at random
                n[0] = int((nct)*random.random())     # select one city
                n[1] = int((nct-1)*random.random())   # select another city, but not the same
                if (n[1] >= n[0]): n[1] += 1   #
                if (n[1] < n[0]): (n[0],n[1]) = (n[1],n[0]) # swap, because it must be: n[0]<n[1]
                nn = (n[0]+nct -n[1]-1) % nct  # number of cities not on the segment n[0]..n[1]
                if nn>=3: break
        
            # We want to have one index before and one after the two cities
            # The order hence is [n2,n0,n1,n3]
            n[2] = (n[0]-1) % nct  # index before n0  -- see figure in the lecture notes
            n[3] = (n[1]+1) % nct  # index after n2   -- see figure in the lecture notes
            
            if Preverse > random.random(): 
                # Here we reverse a segment
                # What would be the cost to reverse the path between city[n[0]]-city[n[1]]?
                de = calc_distance(cities,Route[n[2]],Route[n[1]]) +calc_distance(cities,Route[n[3]],Route[n[0]]) -calc_distance(cities,Route[n[2]],Route[n[0]]) -calc_distance(cities,Route[n[3]],Route[n[1]])
                #de = Distance(R[city[n[2]]],R[city[n[1]]]) + Distance(R[city[n[3]]],R[city[n[0]]]) - Distance(R[city[n[2]]],R[city[n[0]]]) - Distance(R[city[n[3]]],R[city[n[1]]])
                
                if de<0 : # Metropolis
                    accepted += 1
                    dist += de
                    reverse(Route, n)

            else:
                # Here we transpose a segment
                nc = (n[1]+1+ int(random.random()*(nn-1)))%nct  # Another point outside n[0],n[1] segment. See picture in lecture nodes!
                n[4] = nc
                n[5] = (nc+1) % nct
        
                # Cost to transpose a segment
                #de = -Distance(R[city[n[1]]],R[city[n[3]]]) - Distance(R[city[n[0]]],R[city[n[2]]]) - Distance(R[city[n[4]]],R[city[n[5]]])
                de = -calc_distance(cities,Route[n[1]],Route[n[3]]) -calc_distance(cities,Route[n[0]],Route[n[2]]) -calc_distance(cities,Route[n[4]],Route[n[5]]) 
                #de += Distance(R[city[n[0]]],R[city[n[4]]]) + Distance(R[city[n[1]]],R[city[n[5]]]) + Distance(R[city[n[2]]],R[city[n[3]]])
                de += calc_distance(cities,Route[n[0]],Route[n[4]]) +calc_distance(cities,Route[n[1]],Route[n[5]]) +calc_distance(cities,Route[n[2]],Route[n[3]]) 
                
                if de<0: # Metropolis
                    accepted += 1
                    dist += de
                    Route = transpt(Route, n)

                    
            if accepted > maxAccepted: break
            
    return Route
    
def swap_2opt(cities, tour_input):
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
            tabu.append((city_a, city_b))  
            tour[edge_1+1:edge_2+1] = tour[edge_1+1:edge_2+1][::-1]

        if len(improve) > MAX_IMPROVE_LOOP:
            del improve[0]
        improve.append(dist)

        iteration += 1
    tour.append(tour[0])
    return tour                 

def calc_serial_2opt_tour(tsp):
        
    cities = tsp["CITIES"]
    
    #tour = nearest_neighbor(cities,1)
    tour = just_returnTour(cities)


    iteration=200
    STATE=0
    BESTTOURLEN= tour_distance(cities,tour)
    BESTTOUR= copy.copy(tour)
    MAXTABU=5
    TABULIST= [[] for _ in range(MAXTABU)]
    T_DIST = [BESTTOURLEN for _ in range(MAXTABU)]
    for i in xrange (MAXTABU):
       TABULIST[i]= copy.copy(tour)
    TABUPTR=1
    for i in xrange (iteration):
       if STATE >=1:
           index=random.randrange(0,len(TABULIST),1)
           print "Random Generated: %d" %index
           #tour = swap_2opt(TABULIST[index])
           tour = swap_2opt_SA(cities,TABULIST[index],T_DIST[index])
       else:           
           #tour = swap_2opt(BESTTOUR)
           tour = swap_2opt_SA(cities,BESTTOUR,BESTTOURLEN)
       total_dist = tour_distance(cities,tour)
       if total_dist < BESTTOURLEN:
          BESTTOURLEN = total_dist
          BESTTOUR =  copy.copy(tour)
          if TABUPTR < MAXTABU:
             TABULIST[TABUPTR]= copy.copy(tour)
             T_DIST[TABUPTR]= total_dist
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


    BESTTOUR.append(BESTTOUR[0])  # loop back to starting point
    """
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

    """
    print BESTTOURLEN
    #pprint(tsp)         
    #cities = tsp["CITIES"]
    #f = lambda k: 'x: {x} y: {y}'.format(x=k.coord_tuple()[0], y=k.coord_tuple()[1])
    #strs = map(f, cities)
    #pprint(strs)
    return BESTTOUR


def rough_chunk(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out


cities = []

def localsearch(path, proc, queue):
    if len(path) > 3:
        # first and last element must not CHANGE

        path = [path[0]] + list(reversed(path[1:len(path)-1])) + [path[len(path)-1]]
        
    queue.put([proc, path]) # mark the sub tour using processor id


def calc_parallel_2opt_tour(tsp):

    THREADS = 8
    MAX_ITER = 1

    pool = Pool(processes=THREADS)
    cities = tsp["CITIES"]

    tour = range(len(cities))
    tour.append(tour[0])                    # make path into a tour
    
    dist = tour_distance(cities, tour)
    best_dist = dist
    best_tour = tour

    for i in xrange(MAX_ITER):
        new_tour = best_tour
        # rotate new_tour by chunk_sz/2
        new_tour = new_tour[:len(new_tour)-1]
        new_tour = new_tour[chunk_sz/2:] + new_tour[:chunk_sz/2]
        new_tour.append(new_tour[0])

        # split new_tour by THREADS and pass to localsearch()
        splits = rough_chunk(new_tour, THREADS)
        print splits
        queue = Queue()
        procs = []
        for m in xrange(len(splits)):
            # mark the subtour using processor id
            p = Process(target=localsearch, args=(splits[m], m, queue,)) 
            p.Daemon = True
            procs.append(p)
            p.start()

        # merge the collected paths
        for p in procs:
            p.join()
        queue.put('Q_END')        
        new_tour = [None] * THREADS
        for i in iter(queue.get, 'Q_END'):
            new_tour[i[0]] = i[1]
        new_tour = [city for subt in new_tour for city in subt] 

        # replace best with current if better
        dist = tour_distance(cities, new_tour)
        if dist < best_dist:
            best_dist = dist    
            best_tour = new_tour

    return (best_dist, best_tour)


