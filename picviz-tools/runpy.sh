LIBPICVIZPATH=../libpicviz/
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

if [ "$1" == "debug" ]
then
    PICVIZ_PARSERS_DIR=$LIBPICVIZPATH/plugins/parsers PYTHONPATH=.:$LIBPICVIZPATH/bindings/python-ctypes/:$PYTHONPATH LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ gdb python
else
    PICVIZ_PARSERS_DIR=$LIBPICVIZPATH/plugins/parsers PYTHONPATH=.:$LIBPICVIZPATH/bindings/python-ctypes/:$PYTHONPATH LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ python $@
fi
