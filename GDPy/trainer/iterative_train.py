#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import json
import logging
from pathlib import Path
from typing import NoReturn

from GDPy.trainer.train_potential import read_dptrain_json
from GDPy.sampler.sample_main import Sampler

MAXITER = 100

class Progress():

    def __init__():
        pass

class IterativeTrainer():
    """"""

    def __init__(self, inputs: str) -> NoReturn:
        """"""
        inputs_path = Path(inputs)
        main_json = inputs_path / 'main.json'
        with open(main_json, 'r') as fopen:
            main_dict = json.load(fopen)
        self.main_dict = main_dict

        # check few working directories
        main_directory = main_dict.get('main_directory', None) 
        if main_directory is None:
            raise ValueError('main directory is must')
        else:
            main_directory = Path(main_directory)
            self.main_directory = main_directory
            if main_directory.exists():
                self.restart()
            else:
                self.initialise()

        return
    
    def initialise(self) -> NoReturn:
        """"""
        # create the new main dir
        self.main_directory.mkdir()

        return

    def set_logger(self):
        # initialise logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        main_logfile_path = self.main_directory / 'log.txt'
        fh = logging.FileHandler(filename=main_logfile_path, mode='w')

        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        return logger

    def restart(self) -> NoReturn:
        """restart from old dir"""
        self.logger = self.set_logger()
        # begin!
        self.logger.info(
            '\nRestart at %s\n', 
            time.asctime( time.localtime(time.time()) )
        )

        # check database
        main_database = self.main_directory / 'database'
        if main_database.exists():
            self.main_database = main_database
            self.logger.info('\nCheck Database')
        else:
            raise ValueError('database is destroyed.')

        return
    
    def run(self) -> NoReturn:
        # run iterative training
        stages = ['trainer', 'sampler', 'labeler']
        for it in range(MAXITER):
            for stage in stages:
                self.irun(it, stage)

    
        return

    def irun(self, cur_iter, stage, step=0):
        """"""
        iter_directory = self.main_directory / ('it-'+str(cur_iter).zfill(4))
        if stage == 'trainer':
            read_dptrain_json(iter_directory, self.main_database, self.main_dict)
        elif stage == 'sampler':
            print('sampler...')
            scout = Sampler(main_dict)
            if step == 0:
                print('make sample dirs...')
                # sampler_main(iter_directory, self.main_database, self.main_dict)
                scout.create(iter_directory)
            elif step == 1:
                print('collect')
                scout.collect(iter_directory)
                # collect_sample_data(iter_directory, self.main_database, self.main_dict)
            else:
                pass
        elif stage == 'labeler':
            pass                    
        else:
            pass
        
        return 

def iterative_train(inputs):
    return


if __name__ == '__main__':
    inputs = '/users/40247882/repository/GDPy/templates/inputs'
    it = IterativeTrainer(inputs)
    it.irun(11, 'trainer')
    #it.irun(5, 'sampler', 1)
    pass
