import multiprocessing as mp
import time
import random

def heavy_work(id, time_to_sleep):
    print("Starting heavy work for id", id)
    time.sleep(time_to_sleep)
    print("Finished heavy work for id", id)
    return str(id) + " " + str(time_to_sleep)





