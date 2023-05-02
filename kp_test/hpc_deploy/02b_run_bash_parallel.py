import argparse
import multiprocessing
import os
import sys
from joblib import Parallel, delayed, parallel_backend
import stopit

def run_code(i):
    cli = f"timeout 30600 python 02_run_experiment.py --EXP_TITLE kp_test --WORKER_INDEX {i}"
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)

with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
    Parallel()(delayed(run_code)(i) for i in range(0,10800))