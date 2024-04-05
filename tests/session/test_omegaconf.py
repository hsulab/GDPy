#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataclasses
import omegaconf
from omegaconf import OmegaConf

from ase.io import read, write

from GDPy.core.register import import_all_modules_for_register
from GDPy.core.session import create_node, create_op_instance, Session

from GDPy.builder.interface import build

from GDPy.utils.command import CustomTimer

import_all_modules_for_register()

@dataclasses.dataclass
class MyConfig:

    port: int = 80
    hist: str = "localhost"

#conf = omegaconf.OmegaConf.structured(MyConfig)
#print(omegaconf.OmegaConf.to_yaml(conf))

def xxxx():
    OmegaConf.register_new_resolver(
        "structure", lambda x: read(x, ":")
    )

    cfg = OmegaConf.create(
       {
           "plans": {
               "A": "plan A",
               "B": "plan B",
           },
           "selected_plan": "A",
           "plan": "${plans[${selected_plan}]}",
           "xxx": "${structure:/users/40247882/projects/dataset/cof/CH4-CH4-molecule/init.xyz}"
       }
    )

    print(OmegaConf.to_yaml(cfg))
    print(cfg.plan)
    print(cfg.xxx)

def sdasdsa_main():
    """"""
    import json
    def read_json(input_file):
        with open(input_file, "r") as fopen:
            input_dict = json.load(fopen)

        return input_dict

    OmegaConf.register_new_resolver(
        "read_json", read_json #lambda x: read_json(x)
    )
    conf = OmegaConf.load("./test_inputs/dptrainer.yaml")
    print(conf.potential.trainer.config)
    print("params: ", conf.potential.params)
    print(OmegaConf.to_yaml(conf))

    return


def main():
    """"""
    OmegaConf.register_new_resolver(
        "gdp_v", lambda x: create_node("xxx", OmegaConf.to_object(x)),
        use_cache=False
    )
    OmegaConf.register_new_resolver(
        "gdp_o", lambda x: create_op_instance("xxx", **OmegaConf.to_object(x))
    )
    OmegaConf.register_new_resolver(
        "structure", lambda x: build(*[create_node("xxx", {"type": "builder", "method": "molecule", "filename": x})])
    )

    conf = OmegaConf.load("../assets/inputs/minimal/md.yaml")
    #print(conf.variables.worker_nvt)
    #print(conf.sessions.md.nvtw_1)
    
    #for key, value in conf.variables.items():
    #    print(key, value)

    #print(conf.variables.worker_nvt.potential)
    #print(type(conf.variables.worker_nvt.potential))

    #container = OmegaConf.to_object(conf.variables.worker_nvt)
    #print(container)

    #print(conf.sessions.md.nvtw_1.inputs)

    #container = OmegaConf.to_object(conf.sessions.md.nvtw_1)
    #print(container)

    #container = create_op_instance("xxx", **OmegaConf.to_object(conf.sessions.md.nvtw_1))
    #print(container)

    # - test session


    return


if __name__ == "__main__":
    """"""
    with CustomTimer() as ct:
        main()
    ...