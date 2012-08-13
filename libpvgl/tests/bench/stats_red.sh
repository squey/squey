#§bin/bash

# \file stats_red.sh
#
# Copyright (C) Picviz Labs 2010-2012

for i in $(seq 0 1024); do echo -n "$i "; ./nred $((i*1024)) 5000000 |cut -d'|' -f1 |cut -d'/' -f5 |cut -d' ' -f1; done
