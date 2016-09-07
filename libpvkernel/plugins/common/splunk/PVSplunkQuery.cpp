/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVSplunkQuery.h"

PVRush::PVSplunkQuery::PVSplunkQuery(PVSplunkInfos const& infos,
                                     QString const& query,
                                     QString const& query_type)
    : _infos(infos), _query(query), _query_type(query_type)
{
}

bool PVRush::PVSplunkQuery::operator==(const PVInputDescription& other) const
{
	PVSplunkQuery const* other_query = dynamic_cast<PVSplunkQuery const*>(&other);
	if (!other_query) {
		return false;
	}
	return _infos == other_query->_infos && _query == other_query->_query &&
	       _query_type == other_query->_query_type;
}

QString PVRush::PVSplunkQuery::human_name() const
{
	return QString("Splunk");
}

void PVRush::PVSplunkQuery::serialize_write(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Serialize Splunk information.");
	so.attribute("query", _query);
	so.attribute("query_type", _query_type);
	so.object("server", _infos);
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVSplunkQuery::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Searching for Splunk informations.");
	QString query;
	so.attribute("query", query);
	QString query_type;
	so.attribute("query_type", query_type);
	PVSplunkInfos infos;
	so.object("server", infos);
	return std::unique_ptr<PVSplunkQuery>(new PVSplunkQuery(infos, query, query_type));
}

void PVRush::PVSplunkQuery::save_to_qsettings(QSettings& settings) const
{
	settings.setValue("host", _infos.get_host());
	settings.setValue("port", _infos.get_port());
	settings.setValue("query", _query);
	settings.setValue("query_type", _query_type);
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVSplunkQuery::load_from_qsettings(const QSettings& settings)
{
	PVSplunkInfos infos;
	infos.set_host(settings.value("host").toString());
	infos.set_port(settings.value("port").toInt());
	QString query(settings.value("query").toString());
	QString query_type(settings.value("query_type").toString());
	return std::unique_ptr<PVSplunkQuery>(new PVSplunkQuery(infos, query, query_type));
}
