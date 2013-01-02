/**
 * \file PVArcsightInfos.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVDBINFOS_FILE_H
#define PVDBINFOS_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <QString>

namespace PVRush {

class PVArcsightInfos
{
	friend class PVCore::PVSerializeObject;
public:
	PVArcsightInfos();
	PVArcsightInfos(PVArcsightInfos const& infos);
	PVArcsightInfos(QString const& host, uint16_t port, QString const& username, QString const& password);

public:
	void set_host(QString const& host) { _host = host; }
	void set_username(QString const& username) { _username = username; }
	void set_password(QString const& password) { _password = password; }
	void set_port(uint16_t port) { _port = port; }

	QString const& get_host() const { return _host; }
	QString const& get_username() const { return _username; }
	QString const& get_password() const { return _password; }
	uint16_t get_port() const { return _port; }

	inline bool operator==(PVArcsightInfos const& o) const
	{ return _host == o._host &&
	         _username == o._username &&
	         _password == o._password &&
		 _port == o._port; }

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

protected:
	QString _host;
	QString _username;
	QString _password;
	uint16_t _port;
};

}

#endif
