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
	so.attribute_write("login", _login);
	so.attribute_write("importer", _importer);
	so.attribute_write("password", _password);
}

PVRush::PVElasticsearchInfos
PVRush::PVElasticsearchInfos::serialize_read(PVCore::PVSerializeObject& so)
{
	PVRush::PVElasticsearchInfos info;
	info._host = so.attribute_read<QString>("host");
	info._port = so.attribute_read<uint16_t>("port");
	info._index = so.attribute_read<QString>("index");
	info._login = so.attribute_read<QString>("login");
	info._importer = so.attribute_read<QString>("importer");
	info._password = so.attribute_read<QString>("password");
	return info;
}
