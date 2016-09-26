/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVDBInfos.h"
#include "PVDBQuery.h"

#include <pvkernel/core/PVRecentItemsManager.h>

PVRush::PVDBQuery::PVDBQuery()
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBServ_p infos) : _infos(infos)
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBServ_p infos, QString const& query) : _infos(infos)
{
	set_query(query);
}

PVRush::PVDBQuery::~PVDBQuery()
{
}

bool PVRush::PVDBQuery::operator==(const PVInputDescription& other) const
{
	PVDBQuery& other_query = (PVDBQuery&)other;
	return _infos->get_type() == other_query._infos->get_type() &&
	       _infos->get_host() == other_query._infos->get_host() &&
	       _infos->get_username() == other_query._infos->get_username() &&
	       _infos->get_password() == other_query._infos->get_password() &&
	       _infos->get_options() == other_query._infos->get_options() &&
	       _infos->get_dbname() == other_query._infos->get_dbname() &&
	       _infos->get_port() == other_query._infos->get_port();
}

QSqlQuery PVRush::PVDBQuery::to_query(chunk_index /*start*/, chunk_index /*nelts*/) const
{
	QString querystr = _query /* + QString(" LIMIT %1 OFFSET %2").arg(nelts).arg(start)*/;
	QSqlQuery query(_infos->to_database());
	query.prepare(querystr);
	return query;
}

bool PVRush::PVDBQuery::connect_serv()
{
	return _infos->connect();
}

QString PVRush::PVDBQuery::last_error_serv()
{
	return _infos->last_error();
}

QString PVRush::PVDBQuery::human_name() const
{
	return _infos->database_name();
}

void PVRush::PVDBQuery::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Serialize DB information.");
	so.attribute_write("query", _query);
	_infos->serialize_write(*so.create_object("server"));
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVDBQuery::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Searching for Database informations.");
	QString query = so.attribute_read<QString>("query");
	PVDBInfos infos = PVDBInfos::serialize_read(*so.create_object("server"));
	return std::unique_ptr<PVDBQuery>(
	    new PVDBQuery(std::shared_ptr<PVDBServ>(new PVDBServ(infos)), query));
}

void PVRush::PVDBQuery::save_to_qsettings(QSettings& settings) const
{
	settings.setValue("type", _infos->get_type());
	settings.setValue("host", _infos->get_host());
	settings.setValue("username", _infos->get_username());
	settings.setValue("password", _infos->get_password());
	settings.setValue("options", _infos->get_options());
	settings.setValue("dbname", _infos->get_dbname());
	settings.setValue("port", _infos->get_port());
}

std::unique_ptr<PVRush::PVInputDescription>
PVRush::PVDBQuery::load_from_string(std::vector<std::string> const&)
{
	throw PVRush::BadInputDescription("Incomplete input for DBQuery");
}

std::vector<std::string> PVRush::PVDBQuery::desc_from_qsetting(QSettings const&)
{
	throw PVRush::BadInputDescription("Incomplete input for DBQuery");
}
