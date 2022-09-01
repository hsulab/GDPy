#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.utils.command import parse_input_file


def run_expedition(potter, referee, exp_json):
    # - create exploration
    exp_dict = parse_input_file(exp_json)

    method = exp_dict.get("method", "MD")
    if method == "MD":
        from .md import MDBasedExpedition as exp_cls
    elif method == "GA":
        from .randomSearch import RandomExplorer as exp_cls
    elif method == "adsorbate":
        from .adsorbate import AdsorbateEvolution as exp_cls
    elif method == "reaction":
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