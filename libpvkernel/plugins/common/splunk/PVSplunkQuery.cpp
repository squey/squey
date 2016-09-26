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

void PVRush::PVSplunkQuery::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Serialize Splunk information.");
	so.attribute_write("query", _query);
	so.attribute_write("query_type", _query_type);
	_infos.serialize_write(*so.create_object("server"));
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVSplunkQuery::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Searching for Splunk informations.");

	QString query = so.attribute_read<QString>("query");
	QString query_type = so.attribute_read<QString>("query_type");
	PVSplunkInfos infos = PVSplunkInfos::serialize_read(*so.create_object("server"));

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
PVRush::PVSplunkQuery::load_from_string(std::vector<std::string> const&)
{
	throw PVRush::BadInputDescription("Incomplete input for SplunkQuery");
}

std::vector<std::string> PVRush::PVSplunkQuery::desc_from_qsetting(QSettings const&)
{
	throw PVRush::BadInputDescription("Incomplete input for SplunkQuery");
}
