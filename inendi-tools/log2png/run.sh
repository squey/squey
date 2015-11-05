LIBINENDIPATH=../../libinendi
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

INENDI_DEBUG_LEVEL="DEBUG" INENDI_NORMALIZE_DIR=$LIBINENDIPATH/plugins/normalize INENDI_NORMALIZE_HELPERS_DIR=$LIBINENDIPATH/plugins/normalize-helpers LD_LIBRARY_PATH=$LIBINENDIPATH/src/ $@
