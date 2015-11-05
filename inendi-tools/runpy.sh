LIBINENDIPATH=../libinendi/
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

if [ "$1" == "debug" ]
then
    INENDI_PARSERS_DIR=$LIBINENDIPATH/plugins/parsers PYTHONPATH=.:$LIBINENDIPATH/bindings/python-ctypes/:$PYTHONPATH LD_LIBRARY_PATH=$LIBINENDIPATH/src/ gdb python
else
    INENDI_PARSERS_DIR=$LIBINENDIPATH/plugins/parsers PYTHONPATH=.:$LIBINENDIPATH/bindings/python-ctypes/:$PYTHONPATH LD_LIBRARY_PATH=$LIBINENDIPATH/src/ python $@
fi
