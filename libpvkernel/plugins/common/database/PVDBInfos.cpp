/**
 * \file PVDBInfos.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvbase/general.h>
#include "PVDBInfos.h"
#include <QSqlError>

PVRush::PVDBInfos::PVDBInfos()
{
}

PVRush::PVDBInfos::PVDBInfos(QString const& type, QString const& host, uint16_t port, QString const& username, QString const& password, QString const& dbname, QString const& options)
{
	set_type(type);
	set_host(host);
	set_username(username);
	set_password(password);
	set_options(options);
	set_dbname(dbname);
	set_port(port);
}

QString PVRush::PVDBInfos::database_name() const
{
	return _type + QString("://") + _username + QString("@") + _host + QString(":") + QString::number(_port);
}

PVRush::PVDBServ::PVDBServ(PVDBInfos const& infos)
{
	_host = infos._host;
	_username = infos._username;
	_password = infos._password;
	_options = infos._options;
	_type = infos._type;
	_dbname = infos._dbname;
	_port = infos._port;
}

PVRush::PVDBServ::~PVDBServ()
{
	if (_db.isOpen()) {
		_db.close();
	}
}

bool PVRush::PVDBServ::connect()
{
	if (_db.isOpen()) {
		// Already connected;
		return true;
	}

	PVLOG_DEBUG("Connecting to %s...\n", qPrintable(database_name()));
	_db = QSqlDatabase::addDatabase(_type, database_name());
	_db.setHostName(_host);
	_db.setUserName(_username);
	_db.setPassword(_password);
	_db.setPort(_port);
	_db.setDatabaseName(_dbname);
	_db.setConnectOptions(_options);
	bool ret = _db.open();
	if (ret) {
		PVLOG_DEBUG("Connection successful.\n");
	}
	else {
		PVLOG_DEBUG("Connection error: %s.\n", qPrintable(last_error()));
	}
	return ret;
}

QString PVRush::PVDBServ::last_error() const
{
	return _db.lastError().databaseText() + " " + _db.lastError().driverText() + " " + QString::number(_db.lastError().number());
}

QSqlDatabase PVRush::PVDBServ::to_database()
{
	return _db;
}

void PVRush::PVDBInfos::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("host", _host);
	so.attribute("username", _username);
	so.attribute("password", _password);
	so.attribute("options", _options);
	so.attribute("type", _type);
	so.attribute("dbname", _dbname);
	so.attribute("port", _port);
}
