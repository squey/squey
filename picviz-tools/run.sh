LIBPICVIZPATH=../libpicviz

# \file run.sh
#
# Copyright (C) Picviz Labs 2010-2012

PICVIZ_NORMALIZE_DIR=$LIBPICVIZPATH/plugins/normalize PICVIZ_PARSERS_DIR=$LIBPICVIZPATH/plugins/parsers LD_LIBRARY_PATH=../libpicviz/src/ $@

