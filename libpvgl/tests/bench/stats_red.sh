#§bin/bash
for i in $(seq 0 1024); do echo -n "$i "; ./nred $((i*1024)) 10000000 |cut -d'|' -f1 |cut -d'/' -f5 |cut -d' ' -f1; done
