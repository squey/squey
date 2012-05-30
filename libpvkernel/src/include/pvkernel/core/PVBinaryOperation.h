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
	NOR,
	NAND,
	NXOR,
	LAST_BINOP
};

QString get_binary_operation_name(PVBinaryOperation binop);

}

#endif
