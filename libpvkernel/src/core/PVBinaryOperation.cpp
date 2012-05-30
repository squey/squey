
#include <pvkernel/core/PVBinaryOperation.h>

QString PVCore::get_binary_operation_name(PVCore::PVBinaryOperation binop)
{
	switch (binop) {
	case PVCore::OR:
		return QString("OR");
		break;
	case PVCore::AND:
		return QString("AND");
		break;
	case PVCore::XOR:
		return QString("XOR");
		break;
	case PVCore::NOR:
		return QString("OR NOT");
		break;
	case PVCore::NAND:
		return QString("AND NOT");
		break;
	case PVCore::NXOR:
		return QString("XOR NOT");
		break;
	default:
		return QString();
	}
}
