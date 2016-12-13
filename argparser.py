import argparse

parser = argparse.ArgumentParser(
      description = "Parse TSP files and calculate paths using simple "
                    "algorithms.")

parser.add_argument (
      "-s"
    , "--sequential"
    , action  = "store_true"
    , dest    = "need_sequential_2opt"
    , default = False
    , help    = "calculate distance traveled by sequential 2-opt heuristic"
    )

parser.add_argument (
      "-o"
    , "--openmp"
    , action  = "store_true"
    , dest    = "need_openmp_2opt"
    , default = False
    , help    = "calculate distance traveled by openmp 2-opt heuristic"
    )

parser.add_argument (
      "-g"
    , "--gpgpu"
    , action  = "store_true"
    , dest    = "need_gpu_2opt"
    , default = False
    , help    = "calculate distance traveled by gpgpu 2-opt heuristic"
    )

parser.add_argument (
      "tsp_queue"
    , nargs   = "+"
    , metavar = "PATH"
    , help    = "Path to directory or .tsp file. If PATH is a directory, run "
                "on all .tsp files in the directory."
    )
