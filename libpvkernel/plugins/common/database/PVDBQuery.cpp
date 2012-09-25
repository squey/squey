/**
 * \file PVDBQuery.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVDBInfos.h"
#include "PVDBQuery.h"

#include <pvkernel/core/PVRecentItemsManager.h>

PVRush::PVDBQuery::PVDBQuery()
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBServ_p infos):
	_infos(infos)
{
}

PVRush::PVDBQuery::PVDBQuery(PVDBServ_p infos, QString const& query):
	_infos(infos)
{
	set_query(query);
}

PVRush::PVDBQuery::~PVDBQuery()
{
}

bool PVRush::PVDBQuery::operator==(const PVInputDescription& other) const
{
	PVDBQuery& other_query = (PVDBQuery&) other;
	return _infos->get_type() == other_query._infos->get_type() &&
		   _infos->get_host() == other_query._infos->get_host() &&
		   _infos->get_username() == other_query._infos->get_username() &&
		   _infos->get_password() == other_query._infos->get_password() &&
		   _infos->get_options() == other_query._infos->get_options() &&
		   _infos->get_dbname() == other_query._infos->get_dbname() &&
		   _infos->get_port() == other_query._infos->get_port();
}

QSqlQuery PVRush::PVDBQuery::to_query(chunk_index start, chunk_index nelts) const
{
	QString querystr = _query/* + QString(" LIMIT %1 OFFSET %2").arg(nelts).arg(start)*/;
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

void PVRush::PVDBQuery::serialize_write(PVCore::PVSerializeObject& so)
{
	so.attribute("query", _query);
	so.object("server", *_infos, QString(), false, (PVDBServ*) NULL, false);
}

void PVRush::PVDBQuery::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	QString query;
	so.attribute("query", query);
	set_query(query);
	PVDBInfos infos;
	so.object("server", infos);
	_infos.reset(new PVDBServ(infos));
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

void PVRush::PVDBQuery::load_from_qsettings(const QSettings& settings)
{
	_infos->set_type(settings.value("type").toString());
	_infos->set_host(settings.value("host").toString());
	_infos->set_username(settings.value("username").toString());
	_infos->set_password(settings.value("password").toString());
	_infos->set_options(settings.value("options").toString());
	_infos->set_dbname(settings.value("dbname").toString());
	_infos->set_port(settings.value("port").toInt());
}
