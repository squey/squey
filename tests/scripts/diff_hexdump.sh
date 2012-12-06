#!/bin/bash
#

function cleanup()
{
	rm $tmpfifo1
	rm $tmpfifo2
}

if [ $# -ne 2 ]; then
	echo "Usage: $0 file1 file2" 1>&2
	exit 1
fi

HEXDUMP=$(which hexdump)
if [ $? -ne 0 ]; then
	echo "hexdump hasn't been found. Please install it!" 1>&2
	exit 1
fi

tmpfifo1=$(/bin/mktemp -u)
tmpfifo2=$(/bin/mktemp -u)

/usr/bin/mkfifo -m 600 "$tmpfifo1" || exit 1
trap cleanup EXIT
/usr/bin/mkfifo -m 600 "$tmpfifo2" || exit 1

$HEXDUMP -C $1 >$tmpfifo1 &
PID1=$!
$HEXDUMP -C $2 >$tmpfifo2 &
PID2=$!

/usr/bin/diff -u $tmpfifo1 $tmpfifo2
RET=$?

wait $PID1
wait $PID2

exit $RET
