/**
 * \file PVElasticsearchInfos.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVElasticsearchInfos.h"

#include <pvbase/general.h>

PVRush::PVElasticsearchInfos::PVElasticsearchInfos()
{
}

PVRush::PVElasticsearchInfos::PVElasticsearchInfos(PVElasticsearchInfos const& infos)
{
	_host = infos._host;
	_port = infos._port;
	_index = infos._index;
	_login = infos._login;
	_importer = infos._importer;
	_password = infos._password;
}

void PVRush::PVElasticsearchInfos::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("host", _host);
	so.attribute("port", _port);
	so.attribute("index", _index);
	so.attribute("login", _login);
	so.attribute("importer", _importer);
	so.attribute("password", _password);
}
