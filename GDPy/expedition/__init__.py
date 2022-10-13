#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.utils.command import parse_input_file


def run_expedition(potter, referee, exp_json):
    # - create exploration
    exp_dict = parse_input_file(exp_json)

    method = exp_dict.get("method", "md")
    if method == "md":
        from .md import MDBasedExpedition as exp_cls
    elif method == "evo":
        from .evoSearch import EvolutionaryExpedition as exp_cls
    elif method == "ads": # ads
        from .adsorbate import AdsorbateEvolution as exp_cls
    elif method == "rxn": # rxn
        from .reaction import ReactionExplorer as exp_cls
    elif method == "otf":
        from .online.md import OnlineDynamicsBasedExpedition as exp_cls
    else:
        raise NotImplementedError(f"Method {method} is not supported.")

    scout = exp_cls(exp_dict, potter, referee)
    scout.run("./")

    return


if __name__ == "__main__":
    pass