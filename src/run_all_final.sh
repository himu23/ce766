#!/bin/sh
# steps 8-12 in order (about 1 h on 16 threads; step 7 must have finished)
set -e
python 08_optimise.py
python 09_cmip6_factors.py
python 10_robustness.py        # writes the behavioural GR4J sets needed by step 11
python 11_robust_optimise.py
python 10_robustness.py        # again: now tests the robust policies too
python 12_scenarios_prim.py
