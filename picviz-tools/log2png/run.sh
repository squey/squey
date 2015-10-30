LIBPICVIZPATH=../../libpicviz
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

PICVIZ_DEBUG_LEVEL="DEBUG" PICVIZ_NORMALIZE_DIR=$LIBPICVIZPATH/plugins/normalize PICVIZ_NORMALIZE_HELPERS_DIR=$LIBPICVIZPATH/plugins/normalize-helpers LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ $@
