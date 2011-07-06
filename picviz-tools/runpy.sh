LIBPICVIZPATH=../libpicviz/

if [ "$1" == "debug" ]
then
    PICVIZ_PARSERS_DIR=$LIBPICVIZPATH/plugins/parsers PYTHONPATH=.:$LIBPICVIZPATH/bindings/python-ctypes/:$PYTHONPATH LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ gdb python
else
    PICVIZ_PARSERS_DIR=$LIBPICVIZPATH/plugins/parsers PYTHONPATH=.:$LIBPICVIZPATH/bindings/python-ctypes/:$PYTHONPATH LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ python $@
fi
