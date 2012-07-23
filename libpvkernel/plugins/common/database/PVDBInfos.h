/**
 * \file PVDBInfos.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVDBINFOS_FILE_H
#define PVDBINFOS_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <boost/shared_ptr.hpp>

#include <QSqlDatabase>
#include <QString>

#include "PVDBServ_types.h"

namespace PVRush {

class PVDBQuery;
class PVDBServ;

class PVDBInfos
{
	friend class PVDBServ;
	friend class PVCore::PVSerializeObject;
public:
	PVDBInfos();
	PVDBInfos(QString const& type, QString const& host, uint16_t port, QString const& username, QString const& password, QString const& dbname, QString const& options = QString(""));

public:
	void set_type(QString const& type) { _type = type; }
	void set_host(QString const& host) { _host = host; }
	void set_username(QString const& username) { _username = username; }
	void set_password(QString const& password) { _password = password; }
	void set_options(QString const& options) { _options = options; }
	void set_dbname(QString const& dbname) { _dbname = dbname; }
	void set_port(uint16_t port) { _port = port; }

	QString const& get_type() const { return _type; }
	QString const& get_host() const { return _host; }
	QString const& get_username() const { return _username; }
	QString const& get_password() const { return _password; }
	QString const& get_options() const { return _options; }
	QString const& get_dbname() const { return _dbname; }
	uint16_t get_port() const { return _port; }
	
	// That name *must* be unique accross different databases
	QString database_name() const;

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

protected:
	QString _host;
	QString _username;
	QString _password;
	QString _options;
	QString _type;
	QString _dbname;
	uint16_t _port;
};

class PVDBServ: public PVDBInfos
{
	friend class PVDBQuery;
public:
	typedef PVDBServ_p p_type;
public:
	PVDBServ(PVDBInfos const& infos);
	~PVDBServ();

public:
	bool connect();
	QString last_error() const;

protected:
	QSqlDatabase to_database();

private:
	QSqlDatabase _db;
};


}

#endif
