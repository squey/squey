//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVLogger.h>
#include <QSqlError>

#include "PVDBInfos.h"
#include "pvkernel/core/PVSerializeObject.h"

PVRush::PVDBInfos::PVDBInfos()
= default;

PVRush::PVDBInfos::PVDBInfos(QString const& type,
                             QString const& host,
                             uint16_t port,
                             QString const& username,
                             QString const& password,
                             QString const& dbname,
                             QString const& options)
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
	return _type + QString("://") + _username + QString("@") + _host + QString(":") +
	       QString::number(_port);
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
	if (_type == "QPSQL") {
		_db.setConnectOptions("requiressl=1");
	}
	bool ret = _db.open();
	if (ret) {
		PVLOG_DEBUG("Connection successful.\n");
	} else {
		PVLOG_DEBUG("Connection error: %s.\n", qPrintable(last_error()));
	}
	return ret;
}

QString PVRush::PVDBServ::last_error() const
{
	return _db.lastError().databaseText() + " " + _db.lastError().driverText() + " " +
	       _db.lastError().nativeErrorCode();
}

QSqlDatabase PVRush::PVDBServ::to_database()
{
	return _db;
}

void PVRush::PVDBInfos::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.attribute_write("host", _host);
	so.attribute_write("options", _options);
	so.attribute_write("type", _type);
	so.attribute_write("dbname", _dbname);
	so.attribute_write("port", _port);
}

PVRush::PVDBInfos PVRush::PVDBInfos::serialize_read(PVCore::PVSerializeObject& so)
{
	PVRush::PVDBInfos infos;
	infos._host = so.attribute_read<QString>("host");
	infos._options = so.attribute_read<QString>("options");
	infos._type = so.attribute_read<QString>("type");
	infos._dbname = so.attribute_read<QString>("dbname");
	infos._port = so.attribute_read<uint16_t>("port");
	return infos;
}
