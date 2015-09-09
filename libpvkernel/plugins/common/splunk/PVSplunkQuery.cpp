/**
 * \file
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVSplunkQuery.h"

#include <time.h>

PVRush::PVSplunkQuery::PVSplunkQuery(PVSplunkInfos const& infos, QString const& query, QString const& query_type):
	_infos(infos),
	_query(query),
	_query_type(query_type),
	_start_ms(0),
	_end_ms((int64_t)(time(NULL))*1000)
{
}

PVRush::PVSplunkQuery::~PVSplunkQuery()
{
}

bool PVRush::PVSplunkQuery::operator==(const PVInputDescription& other) const
{
	PVSplunkQuery const* other_query = dynamic_cast<PVSplunkQuery const*>(&other);
	if (!other_query) {
		return false;
	}
	return _infos == other_query->_infos &&
	       _query == other_query->_query &&
	       _query_type == other_query->_query_type ;
}

QString PVRush::PVSplunkQuery::human_name() const
{
	return QString("Splunk");
}

void PVRush::PVSplunkQuery::serialize_write(PVCore::PVSerializeObject& so)
{
	so.attribute("query", _query);
	so.attribute("query_type", _query_type);
	so.object("server", _infos);
}

void PVRush::PVSplunkQuery::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("query", _query);
	so.attribute("query_type", _query_type);
	so.object("server", _infos);
}

void PVRush::PVSplunkQuery::save_to_qsettings(QSettings& settings) const
{
	settings.setValue("host", _infos.get_host());
	settings.setValue("port", _infos.get_port());
	settings.setValue("query", _query);
	settings.setValue("query_type", _query_type);
}

void PVRush::PVSplunkQuery::load_from_qsettings(const QSettings& settings)
{
	_infos.set_host(settings.value("host").toString());
	_infos.set_port(settings.value("port").toInt());
	set_query(settings.value("query").toString());
	set_query_type(settings.value("query_type").toString());
}
