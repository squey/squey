/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include "PVOpcUaInfos.h"

void PVRush::PVOpcUaInfos::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.attribute_write("host", _host);
	so.attribute_write("port", _port);
	so.attribute_write("index", _index);
	so.attribute_write("importer", _importer);
	so.attribute_write("format", _format);
	so.attribute_write("is_format_custom", _is_format_custom);
	so.attribute_write("filter_path", _filter_path);
	// Do not save credential as it is not encrypted
}

PVRush::PVOpcUaInfos PVRush::PVOpcUaInfos::serialize_read(PVCore::PVSerializeObject& so)
{
	PVOpcUaInfos info;
	info._host = so.attribute_read<QString>("host");
	info._port = so.attribute_read<uint16_t>("port");
	info._index = so.attribute_read<QString>("index");
	info._importer = so.attribute_read<QString>("importer");
	info._format = so.attribute_read<QString>("format");
	info._is_format_custom = so.attribute_read<bool>("is_format_custom");
	info._filter_path = so.attribute_read<QString>("filter_path");

	if (so.is_repaired_error()) {
		QStringList v = QString::fromStdString(so.get_repaired_value()).split(";");
		info._password = v[1];
		info._login = v[0];
	} else {
		throw PVCore::PVSerializeReparaibleCredentialError("Can't find credential for OpcUa",
		                                                   so.get_logical_path().toStdString());
	}

	return info;
}
