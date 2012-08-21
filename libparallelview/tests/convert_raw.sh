#!/bin/bash
convert -size 1024x1024 -depth 8 rgba:$1 $1.png
