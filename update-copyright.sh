#!/bin/bash

#for fpath in `find . -name "*.h" -o -name "*.c" -o -name  "*.cpp" -type f`; do
#    awk -i inplace '/#/{p=1}p' "$fpath" # Remove old header
#    copyright-header --add-path "$fpath" --license-file LICENSE --word-wrap 80 --output-dir ./ # Add new header
#done

for fpath in `find . -name "*.css" -type f`; do
    sed -i '/* @copyright (C)/d' "$fpath"
done
