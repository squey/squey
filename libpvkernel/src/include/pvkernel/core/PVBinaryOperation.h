#ifndef PVCORE_PVBINARYOPERATION_H
#define PVCORE_PVBINARYOPERATION_H

#include <QString>

namespace PVCore {

enum PVBinaryOperation
{
	FIRST_BINOP = 0,
	OR = FIRST_BINOP,
	AND,
	XOR,
	OR_NOT,
	AND_NOT,
	XOR_NOT,
	LAST_BINOP
};

QString get_binary_operation_name(PVBinaryOperation binop);

}

#endif
