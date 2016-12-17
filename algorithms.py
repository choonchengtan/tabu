from collections        import deque
from city               import distance, GeoCity, Euc_2D, GeoCoord
from tspparse           import read_tsp_file
from pprint             import pprint
from multiprocessing    import Process, Queue, Pool
from functools          import partial
from itertools          import chain
from pycuda.compiler    import SourceModule
from pycuda             import gpuarray
from math               import ceil, floor 
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

    const int MAX_SZ = 3000;                            // maximum path size, each chunk's MAX
    __shared__ __device__ float _city_xy[MAX_SZ * 2];   // store city x,y as x1,x2,...,xn,y1,y2...,yn
    __shared__ __device__ int _city_xy_sz;
    __shared__ __device__ int _path[MAX_SZ];            // path, not tour
    __shared__ __device__ int _path_sz;
    __shared__ __device__ float _cost_reduct[256];      // shared memory for parallel 256 threads

    // for load balancing triangular matrix for every possible 2-opt swap
    // calculate row from linear index i and matrix size M
    __device__ int row_index(int i,int M) {
        float m = M;
        float row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(float)i - 7) )) / -2;
        if(row == (float)(int)row) 
            row -= 1;
        return (int)row;
    }

    // for load balancing triangular matrix for every possible 2-opt swap
    // calculate col from linear index i and matrix size M
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

        // for every thread, find best edge pair individually 
        // then swap the best pair at thread 0
        const int n = path_sz - 2;
        const int MAX_STEPS = n * (n+1) / 2; 
        int pt1, pt2, c1, c2, c3, c4, tmp, k;
        float oe1, oe2, ne1, ne2;
        for (int q=0; q<1; q++) {                          // apply swap to top candidate in all threads' result - 25 times
            for (int i=0; i<MAX_STEPS; i+=blockDim.x) {     // evaluate blockDim.x possible edge swaps per iteration
                k = i + threadIdx.x; 
                if (k < MAX_STEPS) {
                    pt1 = row_index(k, n);
                    pt2 = col_index(k, n) + 2;
                    if (pt2 < _path_sz-1) {
                        c1 = _path[pt1];
                        c2 = _path[pt1+1];
                        c3 = _path[pt2];
                        c4 = _path[pt2+1];
                        oe1 = sqrtf(powf(_city_xy[c1] - _city_xy[c2], 2) + powf(_city_xy[c1 + _path_sz] - _city_xy[c2 + _path_sz], 2));  
                        oe2 = sqrtf(powf(_city_xy[c3] - _city_xy[c4], 2) + powf(_city_xy[c3 + _path_sz] - _city_xy[c4 + _path_sz], 2));  
                        ne1 = sqrtf(powf(_city_xy[c1] - _city_xy[c3], 2) + powf(_city_xy[c1 + _path_sz] - _city_xy[c3 + _path_sz], 2));  
                        ne2 = sqrtf(powf(_city_xy[c2] - _city_xy[c4], 2) + powf(_city_xy[c2 + _path_sz] - _city_xy[c4 + _path_sz], 2));  
                        _cost_reduct[threadIdx.x] = (oe1+oe2) - (ne1+ne2); 
                        //printf("thread %d pt1,pt2 %d,%d cost_redu %.0f\\n", threadIdx.x, pt1, pt2, _cost_reduct[threadIdx.x]);
                    }
                }
                __syncthreads();

                if (threadIdx.x == 0) {
                    printf("[");
                    for (int z=0; z<_path_sz; z++) 
                        printf("%d ", _path[z]); 
                    printf("]\\n");
                    // find maximum cost reduction
                    float maximum = -1;
                    int idx = -1;
                    for (int j=0; j<blockDim.x; j++) {
                        if (_cost_reduct[j] > 0 && _cost_reduct[j] > maximum) {
                            printf("loop %d thread %d cost %.0f\\n", i, j, _cost_reduct[j]);
                            maximum = _cost_reduct[j];
                            idx = j;
                        }
                        _cost_reduct[j] = 0;
                    }
            
                    // swap edge pairs in shared memory if at least one cost reduction exist 
                    // recalculate the path's index pt1 and pt2 from idx
                    if (idx >= 0) {
                        int kt = i + idx; 
                        pt1 = row_index(kt, n);
                        pt2 = col_index(kt, n) + 2;
                        printf("pt1 %d pt2 %d\\n", pt1, pt2);
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
            printf("[");
            for (int z=0; z<_path_sz; z++) 
                printf("%d ", _path[z]); 
            printf("]\\n");
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
    MAX_ITER = 2

    city_xy = [c.x for c in tsp["CITIES"]] + [c.y for c in tsp["CITIES"]] 
    city_xy_n = numpy.array(city_xy)
    
    #tour_n = numpy.array(range(len(tsp["CITIES"])))
    tour_n = numpy.array(nearest_neighbor(tsp["CITIES"],2))
    tour_n = numpy.append(tour_n, tour_n[0])
    tour_sz = tour_n.size
    print tour_distance(tsp["CITIES"], tour_n)

    # split into at least 2 chunks because gpu_2opt_path() only accept paths not tours
    chunk_count = ceil(tour_sz / 3000.0)
    if chunk_count <= 1:
        chunk_count = 2

    city_xy_g = gpuarray.to_gpu(city_xy_n.astype(numpy.float32))

    # for each iteration, split [0 1 2 3 4 5 6 0]
    # into [0 1 2 3 4],[4 5 6 0]
    # because gpu can only hold 3000 city coords each time
    # optimize each split path in gpu one by one, and collect into results
    # results = [[0 1 3 2 4],[4 6 5 0]]
    # flatten and remove one city in between into [0 1 3 2 4 6 5 0]
    # rotate it by 1/3 [3 2 4 6 5 0 1 3]    
    # repeat again
    for itr in xrange(MAX_ITER):
        splits = numpy.array_split(tour_n, chunk_count)
        results = []
        
        for idx, val in enumerate(splits):
            # split paths overlap one city in between, add that
            if idx < len(splits)-1:
                val = numpy.append(val, splits[idx+1][0])

            path_g = gpuarray.to_gpu(val.astype(numpy.int32))
            gpu_2opt_path(city_xy_g, numpy.int32(city_xy_n.size), path_g, numpy.int32(val.size), block=(THREADS, 1, 1))
            cuda.Context.synchronize()
            tmp = path_g.get()

            # split paths overlap one city in between, remove that
            if idx < len(splits)-1:
                tmp = numpy.delete(tmp, tmp.size-1)

            results.append(tmp)

        tour_n = [city for subt in results for city in subt]    # flatten list
        print tour_distance(tsp["CITIES"], tour_n)
        tour_n = numpy.delete(tour_n,len(tour_n)-1)             # remove loop back city
        tour_n = numpy.roll(tour_n, tour_sz/3)                  # rotate tour
        tour_n = numpy.append(tour_n, tour_n[0])                # add back loop back city

    best_tour = tour_n
    best_dist = tour_distance(tsp["CITIES"], best_tour)

    return (best_dist, best_tour)
