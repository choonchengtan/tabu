from collections import deque
from city        import distance, GeoCity, Euc_2D, GeoCoord
from tspparse    import read_tsp_file
from pprint      import pprint
from multiprocessing import Process, Queue, Pool
from functools   import partial
from itertools   import chain
from pycuda.compiler import SourceModule
from pycuda      import gpuarray
import copy
import random
import pycuda.driver as cuda 
import pycuda.tools
import pycuda.autoinit
import numpy as numpy

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

def calc_sequential_2opt_tour(tsp):
        
    cities = tsp["CITIES"]
    
    #tour = nearest_neighbor(cities,1)
    tour =(cities)


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


def local_search(path, proc, cities, queue):
    # local memory: path, proc
    # shared memory: cities(read only), queue(write only)
    PATH_LEN = len(path)
    if PATH_LEN > 4:
        # first and last element must not CHANGE
        MAX_STEPS = 100 * PATH_LEN  # Monte Carlo steps
        MAX_ACCEPT = 10 * PATH_LEN 
        accepted = 0
        for i in xrange(MAX_STEPS):
            point1 = -1
            point2 = -1
            while point1 >= point2 or point1+1 == point2:       # not same city, point1 < point2 and not adjacent 
                point1 = random.randint(0, PATH_LEN - 2)        # select one city
                point2 = random.randint(0, PATH_LEN - 2)        # select one city

            # todo: add transpose  if rand() > Preverse
            if  calc_distance(cities, path[point1], path[point2])   + calc_distance(cities, path[point1+1], path[point2+1]) < \
                calc_distance(cities, path[point1], path[point1+1]) + calc_distance(cities, path[point2], path[point2+1]): 
                # swap edges
                path[point1+1:point2+1] = path[point1+1:point2+1][::-1]
                accepted = accepted + 1                                     

            if accepted > MAX_ACCEPT:
                break

    queue.put([proc, path]) # mark the sub tour using processor id, proc


def calc_openmp_2opt_tour(tsp):

    THREADS = 4
    MAX_ITER = 10   

    cities = tsp["CITIES"]
    chunk_sz = (len(cities)+1) / THREADS
    tour = range(len(cities))
    #tour = nearest_neighbor(cities, random.randint(0,len(cities)-1))
    tour.append(tour[0])                    # make path into a tour
    
    dist = tour_distance(cities, tour)
    best_dist = dist
    best_tour = tour
    print "nearest neighbor"
    print best_dist

    print "search iteration:"
    for i in xrange(MAX_ITER):

        new_tour = best_tour

        # rotate new_tour by chunk_sz/2 or chunk_sz/3 randomly
        new_tour = new_tour[:len(new_tour)-1]
        cut_point = random.randint(2,3)
        new_tour = new_tour[chunk_sz/cut_point:] + new_tour[:chunk_sz/cut_point]
        new_tour.append(new_tour[0])

        # split new_tour by THREADS 
        splits = rough_chunk(new_tour, THREADS)
        if THREADS == 1:    # need to change tour to path because local_search() accepts path           
            splits[0] = splits[0][:len(splits[0])-1]            
        
        # pass to localsearch()
        queue = Queue() # shared queue among all processors 
        procs = []
        for m in xrange(len(splits)):
            # mark the subtour using processor id, m
            p = Process(target=local_search, args=(splits[m], m, cities, queue,)) 
            p.Daemon = True     # dieing parent thread will terminate this child p
            procs.append(p)
            p.start()

        # merge the collected paths in queue
        for p in procs:
            p.join()
        queue.put('QUEUE_END')                          
        new_tour = [None] * THREADS
        for s in iter(queue.get, 'QUEUE_END'):
            new_tour[s[0]] = s[1]
        new_tour = [city for subt in new_tour for city in subt] # flatten list
        if THREADS == 1: # need to change path back to tour because local_search() return path           
            new_tour.append(new_tour[0])

        # replace best solution with current if better
        dist = tour_distance(cities, new_tour)
        print [i, dist]
        if dist < best_dist:
            best_dist = dist    
            best_tour = new_tour

    return (best_dist, best_tour)


mod_gpu = SourceModule("""

#include <stdio.h>

extern "C" {

    const int MAX_SZ = 3000;
    __shared__ __device__ float _city_xy[MAX_SZ * 2];
    __shared__ __device__ int _city_xy_sz;
    __shared__ __device__ int _path[MAX_SZ];
    __shared__ __device__ int _path_sz;
    __shared__ __device__ float _cost_reduct[256]; 

    // purpose of load balancing triangular matrix for every possible 2-opt swap
    // calculate row and col from linear index i and matrix size M
    __device__ int row_index(int i,int M) {
        float m = M;
        float row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(float)i - 7) )) / -2;
        if(row == (float)(int)row) 
            row -= 1;
        return (int)row;
    }

    __device__ int col_index(int i, int M) {
        int row = row_index(i, M);
        return  i - M * row + row*(row+1) / 2;
    }

    __global__ void gpu_2opt_path(float* city_xy, int city_xy_sz, int* path, int path_sz)
    {
        // copy from device global memory into shared memory     
        if (threadIdx.x == 0) {
            _city_xy_sz = city_xy_sz;
            for (int i=0; i<city_xy_sz; i++) {
                _city_xy[i] = city_xy[i];
            }
            _path_sz = path_sz;
            for (int i=0; i<path_sz; i++) {
                _path[i] = path[i];
            }
        }
        __syncthreads();

        // for every thread, find best (edge pt1 + edge pt2) for all threads
        // then swap at thread 0
        const int n = path_sz - 2;
        const int MAX_STEPS = n * (n+1) / 2; 
        int pt1, pt2, c1, c2, c3, c4, tmp, k;
        float oe1, oe2, ne1, ne2;
        for (int q=0; q<25; q++) {       // swap top 5 candidates for all collected threads' result in _cost_reduct
            for (int i=0; i<MAX_STEPS; i+=blockDim.x) {
                k = i + threadIdx.x; 
                if (k< MAX_STEPS) {
                    pt1 = row_index(k, n);
                    pt2 = col_index(k, n) + 2;
                    c1 = _path[pt1];
                    c2 = _path[pt1+1];
                    c3 = _path[pt2];
                    c4 = _path[pt2+1];
                    oe1 = sqrtf(powf(_city_xy[c1] - _city_xy[c2], 2) + powf(_city_xy[c1 + _path_sz] - _city_xy[c2 + _path_sz], 2));  
                    oe2 = sqrtf(powf(_city_xy[c3] - _city_xy[c4], 2) + powf(_city_xy[c3 + _path_sz] - _city_xy[c4 + _path_sz], 2));  
                    ne1 = sqrtf(powf(_city_xy[c1] - _city_xy[c3], 2) + powf(_city_xy[c1 + _path_sz] - _city_xy[c3 + _path_sz], 2));  
                    ne2 = sqrtf(powf(_city_xy[c2] - _city_xy[c4], 2) + powf(_city_xy[c2 + _path_sz] - _city_xy[c4 + _path_sz], 2));  
                    _cost_reduct[threadIdx.x] = (oe1+oe2) - (ne1+ne2); 
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    float maximum = _cost_reduct[0]; 
                    int idx = -1;
                    for (int j=0; j<blockDim.x; j++) {
                        if (_cost_reduct[j] > 0 && _cost_reduct[j] > maximum) {
                            //printf("cost_reduct iter %d loop %d thread %d cost %.0f\\n", q, i, j, _cost_reduct[j]);
                            maximum = _cost_reduct[j];
                            idx = j;
                        }
                        _cost_reduct[j] = 0;
                    }
                    if (idx >= 0) {
                        int kt = i + idx; 
                        pt1 = row_index(kt, n);
                        pt2 = col_index(kt, n) + 2;
                        for (int b=pt1+1, e=pt2; b < e; b++, e--) {
                            tmp = _path[b];
                            _path[b] = _path[e];
                            _path[e] = tmp; 
                        }            
                    }
                } 
                __syncthreads();
            }
           
        }

        // copy back from shared memory to device global memory
        if (threadIdx.x == 0) {
            for (int i=0; i<path_sz; i++) {
                path[i] = _path[i];
            } 
        }
    }
}

""", no_extern_c=True)

def calc_gpu_2opt_tour(tsp):

    gpu_2opt_path = mod_gpu.get_function("gpu_2opt_path")
    
    THREADS = 256 

    city_xy = [c.x for c in tsp["CITIES"]] + [c.y for c in tsp["CITIES"]] 
    city_xy_n = numpy.array(city_xy)
    tour_tmp = range(len(tsp["CITIES"]))
    #tour_tmp = nearest_neighbor(tsp["CITIES"],1)
    print tour_distance(tsp["CITIES"], tour_tmp)
    
    tour_n = numpy.array(tour_tmp)
    city_xy_g = gpuarray.to_gpu(city_xy_n.astype(numpy.float32))
    tour_g = gpuarray.to_gpu(tour_n.astype(numpy.int32))

    gpu_2opt_path(city_xy_g, numpy.int32(city_xy_n.size), tour_g, numpy.int32(tour_n.size), block=(THREADS, 1, 1))
    cuda.Context.synchronize()

    best_tour = tour_g.get()
    best_tour = numpy.append(best_tour, best_tour[0])
    best_dist = tour_distance(tsp["CITIES"], best_tour)

    return (best_dist, best_tour)
