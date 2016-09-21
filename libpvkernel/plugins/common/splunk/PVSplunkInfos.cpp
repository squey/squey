/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVSplunkInfos.h"

void PVRush::PVSplunkInfos::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.attribute_write("host", _host);
	so.attribute_write("port", _port);
	so.attribute_write("login", _login);
	so.attribute_write("password", _password);
	so.attribute_write("splunk_index", _splunk_index);
	so.attribute_write("splunk_host", _splunk_host);
	so.attribute_write("splunk_sourcetype", _splunk_sourcetype);
}

PVRush::PVSplunkInfos PVRush::PVSplunkInfos::serialize_read(PVCore::PVSerializeObject& so)
{
	PVRush::PVSplunkInfos infos;
	infos._host = so.attribute_read<QString>("host");
	infos._port = so.attribute_read<uint16_t>("port");
	infos._login = so.attribute_read<QString>("login");
	infos._password = so.attribute_read<QString>("password");
	infos._splunk_index = so.attribute_read<QString>("splunk_index");
	infos._splunk_host = so.attribute_read<QString>("splunk_host");
	infos._splunk_sourcetype = so.attribute_read<QString>("splunk_sourcetype");
	return infos;
}
