/**
 * \file PVSplunkInfos.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVSplunkInfos.h"

#include <pvbase/general.h>

PVRush::PVSplunkInfos::PVSplunkInfos()
{
}

PVRush::PVSplunkInfos::PVSplunkInfos(PVSplunkInfos const& infos)
{
	_host = infos._host;
	_port = infos._port;
	_login = infos._login;
	_password = infos._password;
	_splunk_index = infos._splunk_index;
	_splunk_host = infos._splunk_host;
	_splunk_sourcetype = infos._splunk_sourcetype;
}

void PVRush::PVSplunkInfos::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("host", _host);
	so.attribute("port", _port);
	so.attribute("login", _login);
	so.attribute("password", _password);
	so.attribute("splunk_index", _splunk_index);
	so.attribute("splunk_host", _splunk_host);
	so.attribute("splunk_sourcetype", _splunk_sourcetype);
}
