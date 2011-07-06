if [ "$1" == "debug" ]
then
    PICVIZ_PARSERS_DIR=../plugins/parsers PYTHONPATH=$PYTHONPATH:.:../srcpy/ LD_LIBRARY_PATH=../src/ gdb ./$2
    exit 0
fi
echo "Running..."
PICVIZ_PARSERS_DIR=../plugins/parsers PYTHONPATH=$PYTHONPATH:.:../srcpy/ LD_LIBRARY_PATH=../src/ ./$1 $2
echo "[DONE]"
