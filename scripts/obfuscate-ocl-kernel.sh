#!/bin/bash

export LC_ALL=C

WORD_FILE="/tmp/word-list.$$"
SUBST_FILE="/tmp/subst.$$"
DICT_FILE="/tmp/dict.$$"

IN_FILE="$1"
OUT_FILE="$2"

VAR_NAME=`basename "$IN_FILE" .cl`


protected_words()
{
	# C99 keywords
	echo auto break case const continue default do else enum extern for goto if inline register restrict return
	echo sizeof static struct switch typedef union volatile while
	echo _Bool _Complex _Imaginary

	# OCL qualifiers
	echo kernel __kernel
	echo global __global local __local constant __constant
	echo read_only __read_only write_only  __write_only read_write  __read_write

	# types
	echo signed unsigned complex imaginary
	echo bool char uchar short ushort int uint long ulong half float double quad size_t ptrdirr_t intptr_t uintptr_t void

	for I in 2 3 4 8 16
	do
		echo bool$I char$I uchar$I short$I ushort$I int$I uint$I long$I ulong$I half$I float$I double$I quad$I
	done

	echo image2d_t image3d_t sampler_t event_t

	# fields shuffles
	echo {x,y,z,w}

	echo {x,y,z,w}{x,y,z,w}

	echo {x,y,z,w}{x,y,z,w}{x,y,z,w}

	echo {x,y,z,w}{x,y,z,w}{x,y,z,w}{x,y,z,w}

	echo a b c d e f A B C D E F
	echo hi lo
	echo even odd

	# OCL group informations
	echo get_group_id get_local_id get_local_size get_num_groups

	# OCL sync
	echo barrier clamp CLK_LOCAL_MEM_FENCE CLK_GLOBAL_MEM_FENCE

	# actually used operations functions
	echo atomic_min fabs fract

	# it misses lots of operations/convertions functions...
}

(
	# extracting words from the file (ignoring alphanumeric and constants (UPPERCASED)
	sed -e 's,//.*$,,g' "$IN_FILE" | grep -o -E '\w+' | grep -v -E '^[A-Z0-9]' | sort -u

	# generate words list to ignore
	(
		protected_words
		protected_words
	) | tr -s "[[:blank:]]" "\n"
) | sort | uniq -u > "$WORD_FILE"

N=`cat "$WORD_FILE" | wc -l`
L=`cat "$WORD_FILE" | wc -L`
L=`expr $L + 1`

pwgen -1 -0 -N $N $L > "$SUBST_FILE"

paste -d\  "$WORD_FILE" "$SUBST_FILE" > "$DICT_FILE"

TEXT=`sed 's/\([^ ]*\) *\(.*\)/s|\\\\<\1\\\\>|\2|g/' "$DICT_FILE" | sed -f - "$IN_FILE" | cpp -P - | tr -s "[:space:]" ' '`

echo "static const char* ${VAR_NAME}_str = \"$TEXT\";" > "$OUT_FILE"

rm -f /tmp/*.$$
