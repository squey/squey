/**
 * \file
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef __PVSPLUNKINFOS_H__
#define __PVSPLUNKINFOS_H__

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <QString>

namespace PVRush {

class PVSplunkInfos
{
	friend class PVCore::PVSerializeObject;
public:
	PVSplunkInfos();
	PVSplunkInfos(PVSplunkInfos const& infos);

public:
	void set_host(QString const& host) { _host = host; }
	void set_port(uint16_t port) { _port = port; }
	void set_login(QString const& login) { _login = login; }
	void set_password(QString const& password) { _password = password; }

	void set_splunk_index(QString const& index) { _splunk_index = index; }
	void set_splunk_host(QString const& host) { _splunk_host = host; }
	void set_splunk_sourcetype(QString const& sourcetype) { _splunk_sourcetype = sourcetype; }

	QString const& get_host() const { return _host; }
	uint16_t get_port() const { return _port; }
	QString const& get_login() const { return _login; }
	QString const& get_password() const { return _password; }

	QString const& get_splunk_index() const { return _splunk_index; }
	QString const& get_splunk_host() const { return _splunk_host; }
	QString const& get_splunk_sourcetype() const { return _splunk_sourcetype; }

	inline bool operator==(PVSplunkInfos const& o) const
	{
		return _host == o._host &&
			   _port == o._port &&
			   _login == o._login &&
			   _password == o._password &&
			   _splunk_index == o._splunk_index &&
			   _splunk_host == o._splunk_host &&
			   _splunk_sourcetype == o._splunk_sourcetype;
	}

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

protected:
	QString _host;
	uint16_t _port;
	QString _login;
	QString _password;

	QString _splunk_index;
	QString _splunk_host;
	QString _splunk_sourcetype;
};

}

#endif // __PVSPLUNKINFOS_H__
