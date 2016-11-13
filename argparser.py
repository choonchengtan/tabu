import argparse

parser = argparse.ArgumentParser(
      description = "Parse TSP files and calculate paths using simple "
                    "algorithms.")

parser.add_argument (
      "-s"
    , "--serial"
    , action  = "store_true"
    , dest    = "need_serial_2opt"
    , default = False
    , help    = "calculate distance traveled by serial 2-opt heuristic"
    )

parser.add_argument (
      "-r"
    , "--parallel"
    , action  = "store_true"
    , dest    = "need_parallel_2opt"
    , default = False
    , help    = "calculate distance traveled by parallel 2-opt heuristic"
    )

parser.add_argument (
      "-p"
    , "--print-tours"
    , action  = "store_true"
    , dest    = "need_tours_printed"
    , default = False
    , help    = "print explicit tours"
    )

parser.add_argument (
      "tsp_queue"
    , nargs   = "+"
    , metavar = "PATH"
    , help    = "Path to directory or .tsp file. If PATH is a directory, run "
                "on all .tsp files in the directory."
    )
