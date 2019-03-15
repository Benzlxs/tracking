# Copyright 2019, Xuesong LI, (email: benzlee08@gmail.com). All Rights Reserved.

import os
import sys
import shutil
import pickle
import fire
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk', 'protos'))


from google.protobuf import text_format
from hbtk.protos import pipeline_pb2


def tracking(config_path,
             output_dir='somewhere',
             ):
    """ main function for tracking

    we run different detectors and combine them together to improve efficiency
    without decreasing perfromance too much. Detection-->Tracking

    Args:
        config_path:
        output_dir:

    Returns:
        sss

    Raises:
        IOError: An error occurred accesssing hte bigtable. Table of object

    """
    config = pipeline_pb2.TrackingPipeline()
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)

    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker

    # build method
    hbtk_detectors = 
    

    # run method

if __name__=='__main__':
    fire.Fire()


