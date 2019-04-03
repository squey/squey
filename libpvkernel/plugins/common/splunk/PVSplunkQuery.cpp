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
	so.set_current_status("Saving Splunk information...");
	so.attribute_write("query", _query);
	so.attribute_write("query_type", _query_type);
	_infos.serialize_write(*so.create_object("server"));
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVSplunkQuery::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Loading Splunk information...");

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
	settings.setValue("format", _infos.get_format());
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVSplunkQuery::load_from_string(std::vector<std::string> const& vl, bool /*multi_inputs*/)
{
	assert(vl.size() >= 5);

	QString query = QString::fromStdString(vl[0]);
	QString query_type = QString::fromStdString(vl[1]);

	PVSplunkInfos infos;
	infos.set_host(QString::fromStdString(vl[2]));
	infos.set_port(std::stoi(vl[3]));
	infos.set_format(QString::fromStdString(vl[4]));

	if (vl.size() == 7) {
		infos.set_login(QString::fromStdString(vl[5]));
		infos.set_password(QString::fromStdString(vl[6]));
	}

	return std::unique_ptr<PVSplunkQuery>(new PVSplunkQuery(infos, query, query_type));
}

std::vector<std::string> PVRush::PVSplunkQuery::desc_from_qsetting(QSettings const& s)
{
	std::vector<std::string> res = {
	    s.value("query").toString().toStdString(), s.value("query_type").toString().toStdString(),
	    s.value("host").toString().toStdString(), s.value("port").toString().toStdString(),
	    s.value("format").toString().toStdString()};
	return res;
}
