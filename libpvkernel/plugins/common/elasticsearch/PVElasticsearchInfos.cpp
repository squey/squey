/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVElasticsearchInfos.h"

void PVRush::PVElasticsearchInfos::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.attribute_write("host", _host);
	so.attribute_write("port", _port);
	so.attribute_write("index", _index);
	so.attribute_write("importer", _importer);
	// Do not save credential as it is not encrypted
}

PVRush::PVElasticsearchInfos
PVRush::PVElasticsearchInfos::serialize_read(PVCore::PVSerializeObject& so)
{
	PVElasticsearchInfos info;
	info._host = so.attribute_read<QString>("host");
	info._port = so.attribute_read<uint16_t>("port");
	info._index = so.attribute_read<QString>("index");
	info._importer = so.attribute_read<QString>("importer");

	if (so.is_repaired_error()) {
		QStringList v = QString::fromStdString(so.get_repaired_value()).split(";");
		info._password = v[1];
		info._login = v[0];
	} else {
		throw PVCore::PVSerializeReparaibleCredentialError(
		    "Can't find credential for Elasticsearch", so.get_logical_path().toStdString());
	}

	return info;
}
