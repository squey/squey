/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVElasticsearchQuery.h"

PVRush::PVElasticsearchQuery::PVElasticsearchQuery(PVElasticsearchInfos const& infos,
                                                   QString const& query,
                                                   QString const& query_type)
    : _infos(infos), _query(query), _query_type(query_type)
{
}

bool PVRush::PVElasticsearchQuery::operator==(const PVInputDescription& other) const
{
	PVElasticsearchQuery const* other_query = dynamic_cast<PVElasticsearchQuery const*>(&other);
	if (!other_query) {
		return false;
	}
	return _infos == other_query->_infos && _query == other_query->_query &&
	       _query_type == other_query->_query_type;
}

QString PVRush::PVElasticsearchQuery::human_name() const
{
	return QString("elasticsearch");
}

void PVRush::PVElasticsearchQuery::serialize_write(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Serialize ElasticSearch information.");
	so.attribute("query", _query);
	so.attribute("query_type", _query_type);
	so.object("server", _infos);
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVElasticsearchQuery::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Searching for ElasticSearch informations.");
	QString query;
	so.attribute("query", query);
	QString query_type;
	so.attribute("query_type", query_type);
	PVElasticsearchInfos infos;
	so.object("server", infos);
	return std::unique_ptr<PVElasticsearchQuery>(
	    new PVElasticsearchQuery(infos, query, query_type));
}

void PVRush::PVElasticsearchQuery::save_to_qsettings(QSettings& settings) const
{
	settings.setValue("host", _infos.get_host());
	settings.setValue("port", _infos.get_port());
	settings.setValue("index", _infos.get_index());
	settings.setValue("query", _query);
	settings.setValue("query_type", _query_type);
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVElasticsearchQuery::load_from_string(std::string const&)
{
	throw PVRush::BadInputDescription("Incomplete input for ESQuery");
}

std::string PVRush::PVElasticsearchQuery::desc_from_qsetting(QSettings const&)
{
	throw PVRush::BadInputDescription("Incomplete input for ESQuery");
}
