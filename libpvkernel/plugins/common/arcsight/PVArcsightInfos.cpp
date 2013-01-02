/**
 * \file PVArcsightInfos.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvbase/general.h>
#include "PVArcsightInfos.h"

PVRush::PVArcsightInfos::PVArcsightInfos()
{
}

PVRush::PVArcsightInfos::PVArcsightInfos(QString const& host, uint16_t port, QString const& username, QString const& password)
{
	set_host(host);
	set_username(username);
	set_password(password);
	set_port(port);
}

PVRush::PVArcsightInfos::PVArcsightInfos(PVArcsightInfos const& infos)
{
	_host = infos._host;
	_username = infos._username;
	_password = infos._password;
	_port = infos._port;
}

void PVRush::PVArcsightInfos::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("host", _host);
	so.attribute("username", _username);
	so.attribute("password", _password);
	so.attribute("port", _port);
}
