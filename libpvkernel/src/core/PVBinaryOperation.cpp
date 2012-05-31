
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
	case PVCore::OR_NOT:
		return QString("OR NOT");
		break;
	case PVCore::AND_NOT:
		return QString("AND NOT");
		break;
	case PVCore::XOR_NOT:
		return QString("XOR NOT");
		break;
	default:
		return QString();
	}
}
