#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.utils.command import parse_input_file


def run_exploration(pot_manager, exp_json, chosen_step, global_params = None):
    # - create exploration
    exp_dict = parse_input_file(exp_json)

    method = exp_dict.get("method", "MD")
    if method == "MD":
        from .md import MDBasedExpedition
        scout = MDBasedExpedition(pot_manager, exp_dict)
    elif method == "GA":
        from .randomSearch import RandomExplorer
        scout = RandomExplorer(pot_manager, exp_dict)
    elif method == "adsorbate":
        from .adsorbate import AdsorbateEvolution
        scout = AdsorbateEvolution(pot_manager, exp_dict)
    elif method == "reaction":
        from .reaction import ReactionExplorer
        scout = ReactionExplorer(pot_manager, exp_dict)
    else:
        raise ValueError(f"Unknown method {method}")

    # - adjust global params
    print("optional params ", global_params)
    if global_params is not None:
        assert len(global_params)%2 == 0, "optional params must be key-pair"
        for first in range(0, len(global_params), 2):
            print(global_params[first], " -> ", global_params[first+1])
            scout.default_params[chosen_step][global_params[first]] = eval(global_params[first+1])

    # compute
    op_name = "i" + chosen_step
    assert isinstance(op_name, str), "op_nam must be a string"
    op = getattr(scout, op_name, None)
    if op is not None:
        scout.run(op, "./")
    else:
        raise ValueError("Wrong chosen step %s..." %op_name)

    return


if __name__ == "__main__":
    pass