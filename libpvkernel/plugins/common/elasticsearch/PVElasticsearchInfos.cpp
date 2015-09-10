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

void PVRush::PVElasticsearchInfos::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("host", _host);
	so.attribute("port", _port);
	so.attribute("index", _index);
	so.attribute("login", _login);
	so.attribute("importer", _importer);
	so.attribute("password", _password);
}
