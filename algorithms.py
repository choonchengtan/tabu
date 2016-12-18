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

    
def tour_distance(cities, tour):
    total = 0
    for i,j in zip(tour[:-1], tour[1:]):
        total += calc_distance(cities, i, j)
    return total


def nearest_neighbor(cities,start):
    tour = [start]
    remaining = range(len(cities))
    del remaining[start]
    while remaining:
        curr = tour[-1]
        dists = [(lambda p, q: calc_distance(cities,p, q))(p, curr) for p in remaining] 
        indexmin = dists.index(min(dists))  
        tour.append(remaining.pop(indexmin))
    return tour 


def calc_sequential_2opt_tour(tsp):
    THREADS = 1
    MAX_ITER = 10   

    cities = tsp["CITIES"]
    chunk_sz = (len(cities)+1) / THREADS
    tour = range(len(cities))
    #tour = nearest_neighbor(cities, random.randint(0,len(cities)-1))
    tour.append(tour[0])                    # make path into a tour
    
    dist = tour_distance(cities, tour)
    best_dist = dist
    best_tour = tour
    print "initial distance"
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
    print "initial distance"
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

    const int MAX_SZ = 3584;                            // maximum path size, each chunk's MAX, multiple of thread size
    __shared__ __device__ float _city_xy[MAX_SZ * 2];   // store city x,y as x1,x2,...,xn,y1,y2...,yn
    __shared__ __device__ int _city_xy_sz;
    __shared__ __device__ unsigned int _path[MAX_SZ];            // path, not tour
    __shared__ __device__ int _path_sz;
    __shared__ __device__ float _cost_reduct[512];      // shared memory for parallel 512 threads

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
        for (int q=0; q<10; q++) {                          // apply swap to top candidate in all threads' result - 10 times
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
                        oe1 = sqrtf(powf(_city_xy[c1] - _city_xy[c2], 2) + powf(_city_xy[c1 + _city_xy_sz/2] - _city_xy[c2 + _city_xy_sz/2], 2));  
                        oe2 = sqrtf(powf(_city_xy[c3] - _city_xy[c4], 2) + powf(_city_xy[c3 + _city_xy_sz/2] - _city_xy[c4 + _city_xy_sz/2], 2));  
                        ne1 = sqrtf(powf(_city_xy[c1] - _city_xy[c3], 2) + powf(_city_xy[c1 + _city_xy_sz/2] - _city_xy[c3 + _city_xy_sz/2], 2));  
                        ne2 = sqrtf(powf(_city_xy[c2] - _city_xy[c4], 2) + powf(_city_xy[c2 + _city_xy_sz/2] - _city_xy[c4 + _city_xy_sz/2], 2));  
                        _cost_reduct[threadIdx.x] = (oe1+oe2) - (ne1+ne2); 
                    }
                }
                __syncthreads();

                if (threadIdx.x == 0) {
                    // find maximum cost reduction
                    float maximum = -1;
                    int idx = -1;
                    for (int j=0; j<blockDim.x; j++) {
                        if (_cost_reduct[j] > 0 && _cost_reduct[j] > maximum) {
                            //printf("loop %d thread %d cost %.0f\\n", i, j, _cost_reduct[j]);
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
    
    THREADS = 512
    MAX_ITER = 6

    city_xy = [c.x for c in tsp["CITIES"]] + [c.y for c in tsp["CITIES"]] 
    city_xy_n = numpy.array(city_xy)
    city_sz = len(tsp["CITIES"])
    
    tour_n = numpy.array(range(city_sz))
    #tour_n = numpy.array(nearest_neighbor(tsp["CITIES"],2))
    tour_n = numpy.append(tour_n, tour_n[0])
    tour_sz = tour_n.size
    print "initial distance"
    print tour_distance(tsp["CITIES"], tour_n)

    # split into at least 2 chunks because gpu_2opt_path() only accept paths not tours
    chunk_count = ceil(tour_sz / 3584.0)
    if chunk_count <= 1:
        chunk_count = 2

    # for each iteration, split [0 1 2 3 4 5 6 0]
    # into [0 1 2 3 4],[4 5 6 0]
    # because gpu can only hold 3584 city coords and tour each time
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

            # extract partial city coords into city_xy_p             
            city_xy_p = [ city_xy[c] for c in val ] + [ city_xy[c+city_sz] for c in val ]
            city_xy_p = numpy.array(city_xy_p)
            city_xy_g = gpuarray.to_gpu(city_xy_p.astype(numpy.float32))
            
            # remap path to partial city coords
            path_n = numpy.array(xrange(len(val)))
            path_g = gpuarray.to_gpu(path_n.astype(numpy.int32))

            gpu_2opt_path(city_xy_g, numpy.int32(city_xy_p.size), path_g, numpy.int32(path_n.size), block=(THREADS, 1, 1))
            cuda.Context.synchronize()
            tmp = path_g.get()

            # recover original path indexing from partial city coords
            for i, v in enumerate(tmp):
                tmp[i] = val[v]

            # split paths overlap one city in between, remove that
            if idx < len(splits)-1:
                tmp = numpy.delete(tmp, tmp.size-1)

            results.append(tmp)

        tour_n = [city for subt in results for city in subt]    # flatten list
        print tour_distance(tsp["CITIES"], tour_n)
        tour_n = numpy.delete(tour_n,len(tour_n)-1)             # remove loop back city
        cut_point = random.randint(2,3)
        tour_n = numpy.roll(tour_n, tour_sz/cut_point)          # rotate tour
        tour_n = numpy.append(tour_n, tour_n[0])                # add back loop back city

    best_tour = tour_n
    best_dist = tour_distance(tsp["CITIES"], best_tour)

    return (best_dist, best_tour)
