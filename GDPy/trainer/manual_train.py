#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pathlib

from GDPy.trainer.train_potential import read_dptrain_json
from GDPy.sampler.sample_main import Sampler


def manual_train(main_json, niter, stage):
    """"""
    with open(main_json, 'r') as fopen:
        main_dict = json.load(fopen)

    main_directory = pathlib.Path(main_dict['main_directory'])
    iter_directory = main_directory / str('it-'+str(niter).zfill(4))
    print(iter_directory)

    main_database = 'dummy'
    stage, step = stage.split('-')
    step = int(step)

    # start iteration
    if stage == 'trainer':
        read_dptrain_json(iter_directory, main_database, main_dict)
    elif stage == 'sampler':
        print('sampler...')
        scout = Sampler(main_dict)
        if step == 0:
            print('make sample dirs...')
            # sampler_main(iter_directory, main_database, main_dict)
            #scout.icreate('TEST', iter_directory)
            scout.create(iter_directory)
        elif step == 1:
            print('collect')
            # collect_sample_data(iter_directory, main_database, main_dict)
            scout.collect(iter_directory)
        elif step == 2:
            print('select')
            # collect_sample_data(iter_directory, main_database, main_dict)
            scout.select(iter_directory)
        else:
            print('no action...')
            pass
    elif stage == 'labeler':
        pass                    
    else:
        pass


    return


if __name__ == '__main__':
    pass