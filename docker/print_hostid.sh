#!/bin/bash

source "resources/.env.conf"

docker exec -it inendi-inspector ${INSTALL_PATH}/rlmutil rlmhostid -q
