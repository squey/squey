/**
 * \file PVElasticsearchInfos.h
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef PVELASTICSEARCHINFOS_H
#define PVELASTICSEARCHINFOS_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <QString>

namespace PVRush {

class PVElasticsearchInfos
{
	friend class PVCore::PVSerializeObject;
public:
	PVElasticsearchInfos();
	PVElasticsearchInfos(PVElasticsearchInfos const& infos);

public:
	void set_host(QString const& host) { _host = host; }
	void set_port(uint16_t port) { _port = port; }
	void set_index(QString const& index) { _index = index; }
	void set_login(QString const& login) { _login = login; }
	void set_password(QString const& password) { _password = password; }

	QString const& get_host() const { return _host; }
	uint16_t get_port() const { return _port; }
	QString const& get_index() const { return _index; }
	QString const& get_login() const { return _login; }
	QString const& get_password() const { return _password; }

	inline bool operator==(PVElasticsearchInfos const& o) const
	{
		return _host == o._host &&
			   _port == o._port &&
			   _index == o._index &&
			   _login == o._login &&
			   _password == o._password;
	}

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

protected:
	QString _host;
	uint16_t _port;
	QString _index;
	QString _login;
	QString _password;
};

}

#endif
