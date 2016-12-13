#!/usr/bin/env python
import time

from argparser  import parser
from tspparse   import read_tsp_file
from algorithms import ( calc_sequential_2opt_tour
                       , calc_openmp_2opt_tour 
                       , calc_gpu_2opt_tour )

from glob    import iglob
from os.path import isfile, isdir, join, exists

def glean_tsp_files(path_arg_list):
    for path_arg in path_arg_list:

        if isdir(path_arg):
            for filepath in iglob(join(path_arg,"*.tsp")):
                yield filepath

        elif isfile(path_arg) & str(path_arg).endswith(".tsp"):
            yield path_arg

        elif isfile(path_arg) & (not path_arg.endswith(".tsp")):
            print("Can't open file ``{0}'': not a .tsp file".format(path_arg))

        elif exists(path_arg):
            print("Path {0} is neither a file nor a directory".format(path_arg))

        else:
            print("Path {0} does not exist".format(path_arg))

def print_results_from_tsp_path(call_args, tsp_path):
    tsp = read_tsp_file(tsp_path)
    print("")
    print("Problem: {}".format(tsp_path))

    if call_args.need_sequential_2opt:
        print("Sequential:")
        print("")
        print("Tour: {}"
             . format(calc_sequential_2opt_tour(tsp)))

    if call_args.need_openmp_2opt:
        print("Open MP:")
        print("")
        result = calc_openmp_2opt_tour(tsp)
        print("Tour Length = {} | Cities = {}" . format(result[0], result[1]))

    if call_args.need_gpu_2opt:
        print("GPGPU:")
        print("")
        result = calc_gpu_2opt_tour(tsp)
        print("Tour Length = {} | Cities = {}" . format(result[0], result[1]))

    del(tsp)

def assignment():
    call_args = parser.parse_args()
    for tsp_path in glean_tsp_files(call_args.tsp_queue):
        print_results_from_tsp_path(call_args,tsp_path)

if __name__ == "__main__":

    f = open('tsplog', 'w')
    tic = time.time()
    assignment()
    toc = time.time()
    print ('Processing time: %r\n' % (toc - tic))
    print("")
    f.write('Processing time: %r\n' % (toc - tic))
    f.closed

