#!/bin/bash

echo "Test plotting CSV"
diff lowlevel-plotted.ref.csv lowlevel-plotted.csv
if [ $? != 0 ]
then
    echo "Difference between plotted CSV"
    exit 1
fi


