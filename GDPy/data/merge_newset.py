#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from ase.io import read, write


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num", type=int,
        required=True,
        help="number of groups"
    )
    args = parser.parse_args()

    # read groups
    cluster_path = Path("./groups/")
    group_files = []
    #for p in cluster_path.iterdir():
    #    if (p.name).startswith('group'):
    #        group_files.append(p)
    #group_files.sort()
    cluster_size = args.num # without noise
    for i in range(cluster_size+1):
        group_name = cluster_path / ("group-"+str(i)+".xyz")
        cur_gname = cluster_path / ("group-"+str(i)+"_cured.xyz")
        if group_name.exists():
            if cur_gname.exists():
                print("used cur-ed xyz file for cluster group ", i)
                group_files.append(cur_gname)
            else:
                print("used original xyz file for cluster group ", i)
                group_files.append(group_name)
        else:
            print("cannot find cluster group ", i)

    # merge all structures
    frames = []
    for gf in group_files:
        frames.extend(
            read(gf, ":")
        )

    write("reduced.xyz", frames)
