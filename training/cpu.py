import os
import sys
import sys
import os
import psutil
import numpy as np
import time
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Suppress other TensorFlow logs


def set_cores(n_cores):
    """
    Set the CPU cores to use for the current process.

    Parameters
    ----------
    n_cores : int
        n_cores: Number of cores to use.
    """

    # Get the least active cores
    least_active_cores = get_least_active_cores(n_cores)

    # Set the least active cores to use
    limit_cpu_cores(least_active_cores)


def limit_cpu_cores(cores_to_use):

    num_cores = len(cores_to_use)

    pid = os.getpid()  # the current process

    available_cores = list(range(psutil.cpu_count()))
    # selected_cores = available_cores[:num_cores]
    selected_cores = []
    for ii in cores_to_use:
        if ii in available_cores:
            selected_cores.append(ii)

    os.sched_setaffinity(pid, selected_cores)

    # Limit the number of threads used by different libraries
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)


def get_least_active_cores(num_cores, num_readings=10):

    # Get CPU usage for each core for multiple readings
    cpu_usage_readings = []
    for ii in range(num_readings):
        cpu_usage_readings.append(psutil.cpu_percent(percpu=True))
        time.sleep(0.05)

    # Calculate the average CPU usage for each core
    avg_cpu_usage = [sum(usage) / num_readings for usage in zip(*cpu_usage_readings)]

    # Create a list of tuples (core_index, avg_cpu_usage)
    core_usage_tuples = list(enumerate(avg_cpu_usage))

    # Sort the list based on average CPU usage
    sorted_cores = sorted(core_usage_tuples, key=lambda x: x[1])

    # Get the first 'num_cores' indices (least active cores)
    least_active_cores = [index for index, _ in sorted_cores[:num_cores]]

    return least_active_cores

