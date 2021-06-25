#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
Potential Manager
deals with various machine learning potentials
"""

import abc


class AbstractPotential(abc.ABC):
    """
    Create various potential instances
    """

    def __init__(self):
        """
        """

        self.uncertainty = None # uncertainty estimation method

        return
    
    @abc.abstractmethod
    def generate_calculator(self):
        """
        valid_dict = {}

        self.nmodels = len(valid_dict['potential']['model'])
        calc = DP(
            type_dict = valid_dict['potential']['type_map'],
            model = valid_dict['potential']['model'] # check number of models
        """
        

        return 

class DPManager(AbstractPotential):

    name = 'DP'

    def __init__(self, models, *args, **kwargs):

        print(models)

        return
    
    def generate_calculator(self):

        return

class NPManager(AbstractPotential):

    name = 'NP'

    def __init__(self, what):

        print(what)

        return

    def generate_calculator(self):

        return


if __name__ == '__main__':
    pass