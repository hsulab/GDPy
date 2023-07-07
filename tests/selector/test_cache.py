#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile

from GDPy.selector.selector import save_cache, load_cache

temp_info = """#      index    confid      step    natoms           ene        aene      maxfrc       score
         3,1         1        10        40     -283.3720     -7.0843      2.2355         nan
         1,1         1        10        40     -283.2048     -7.0801      2.4317         nan
         1,7         1        70        40     -285.4730     -7.1368      2.2214         nan
         1,4         1        40        40     -284.0159     -7.1004      2.3146         nan
         0,1         0        10        40     -283.1768     -7.0794      2.3500         nan
         0,7         0        70        40     -285.4587     -7.1365      2.0297         nan
        1,10         1       100        40     -285.8283     -7.1457      1.3351         nan
        3,10         1       100        40     -285.0987     -7.1275      2.2206         nan
         3,7         1        70        40     -284.3956     -7.1099      2.2663         nan
random_seed 6094
"""

def test_load_cache():
    """"""
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as fopen:
            fopen.write(temp_info)

        markers = load_cache(tmp.name)
    
    #assert raw_unmasks == [[0, [1, 7]], [1, [1, 4, 7, 10]], [3,[1, 7, 10]]]
    assert markers == [[3,1],[1,1],[1,7],[1,4],[0,1],[0,7],[1,10],[3,10],[3,7]]


if __name__ == "__main__":
    ...