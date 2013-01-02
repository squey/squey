/**
 * \file PVArcsightQuery.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVArcsightQuery.h"

#include <pvkernel/core/PVRecentItemsManager.h>

#include <time.h>

PVRush::PVArcsightQuery::PVArcsightQuery():
	_start_ms(0),
	_end_ms((int64_t)(time(NULL))*1000)
{
}

PVRush::PVArcsightQuery::PVArcsightQuery(PVArcsightInfos const& infos):
	_infos(infos),
	_start_ms(0),
	_end_ms((int64_t)(time(NULL))*1000)
{
}

PVRush::PVArcsightQuery::PVArcsightQuery(PVArcsightInfos const& infos, QString const& query):
	_infos(infos),
	_query(query),
	_start_ms(0),
	_end_ms((int64_t)(time(NULL))*1000)
{
}

PVRush::PVArcsightQuery::~PVArcsightQuery()
{
}

bool PVRush::PVArcsightQuery::operator==(const PVInputDescription& other) const
{
	PVArcsightQuery const* other_query = dynamic_cast<PVArcsightQuery const*>(&other);
	if (!other_query) {
		return false;
	}
	return _infos == other_query->_infos &&
	       _query == other_query->_query;
}

QString PVRush::PVArcsightQuery::human_name() const
{
	return QString("human name");
}

void PVRush::PVArcsightQuery::serialize_write(PVCore::PVSerializeObject& so)
{
	so.attribute("query", _query);
	so.object("server", _infos);
}

void PVRush::PVArcsightQuery::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("query", _query);
	so.object("server", _infos);
}

void PVRush::PVArcsightQuery::save_to_qsettings(QSettings& settings) const
{
	settings.setValue("host", _infos.get_host());
	settings.setValue("username", _infos.get_username());
	settings.setValue("password", _infos.get_password());
	settings.setValue("port", _infos.get_port());
	settings.setValue("query", _query);
}

void PVRush::PVArcsightQuery::load_from_qsettings(const QSettings& settings)
{
	_infos.set_host(settings.value("host").toString());
	_infos.set_username(settings.value("username").toString());
	_infos.set_password(settings.value("password").toString());
	_infos.set_port(settings.value("port").toInt());
	set_query(settings.value("query").toString());
}
