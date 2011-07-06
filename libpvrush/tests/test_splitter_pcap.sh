#!/bin/bash
#

DIR="test-files/splitters/pcap"

for f in $DIR/*.pcap; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	./diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_pcap "$INPUT" 6000 || (echo "Failed" && exit 1)
done
