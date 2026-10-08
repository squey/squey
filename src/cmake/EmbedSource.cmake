# Writes a header declaring a string that holds a source file as it is written: the
# OpenCL kernel, which the application builds at run time, for the device it finds.
#
# Usage: cmake -DINPUT=<source> -DOUTPUT=<header> -DNAME=<variable> -P EmbedSource.cmake

file(READ "${INPUT}" source)

# A raw string literal takes the source as it is, up to its closing sequence.
set(delimiter "source")
string(FIND "${source}" ")${delimiter}\"" collision)
if (NOT collision EQUAL -1)
	message(FATAL_ERROR "${INPUT} holds \")${delimiter}\"\", which would end the string early")
endif()

file(WRITE "${OUTPUT}" "static const char* ${NAME} = R\"${delimiter}(${source})${delimiter}\";\n")
