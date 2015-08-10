/**
 * \file PVElasticsearchQuery.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVElasticsearchQuery.h"

#include <time.h>

PVRush::PVElasticsearchQuery::PVElasticsearchQuery(PVElasticsearchInfos const& infos, QString const& query, QString const& query_type):
	_infos(infos),
	_query(query),
	_start_ms(0),
	_end_ms((int64_t)(time(NULL))*1000)
{
}

PVRush::PVElasticsearchQuery::~PVElasticsearchQuery()
{
}

bool PVRush::PVElasticsearchQuery::operator==(const PVInputDescription& other) const
{
	PVElasticsearchQuery const* other_query = dynamic_cast<PVElasticsearchQuery const*>(&other);
	if (!other_query) {
		return false;
	}
	return _infos == other_query->_infos &&
	       _query == other_query->_query &&
	       _query_type == other_query->_query_type ;
}

QString PVRush::PVElasticsearchQuery::human_name() const
{
	return QString("elasticsearch");
}

void PVRush::PVElasticsearchQuery::serialize_write(PVCore::PVSerializeObject& so)
{
	so.attribute("query", _query);
	so.attribute("query_type", _query_type);
	so.object("server", _infos);
}

void PVRush::PVElasticsearchQuery::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("query", _query);
	so.attribute("query_type", _query_type);
	so.object("server", _infos);
}

void PVRush::PVElasticsearchQuery::save_to_qsettings(QSettings& settings) const
{
	settings.setValue("host", _infos.get_host());
	settings.setValue("port", _infos.get_port());
	settings.setValue("index", _infos.get_index());
	settings.setValue("query", _query);
	settings.setValue("query_type", _query_type);
}

void PVRush::PVElasticsearchQuery::load_from_qsettings(const QSettings& settings)
{
	_infos.set_host(settings.value("host").toString());
	_infos.set_port(settings.value("port").toInt());
	_infos.set_index(settings.value("index").toString());
	set_query(settings.value("query").toString());
	set_query_type(settings.value("query_type").toString());
}
