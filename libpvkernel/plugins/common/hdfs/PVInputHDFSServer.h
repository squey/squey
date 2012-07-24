/**
 * \file PVInputHDFSServer.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTHDFSSERVER_FILE_H
#define PVINPUTHDFSSERVER_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <boost/shared_ptr.hpp>

#include <hadoop/hdfs.h>


namespace PVRush {

class PVInputHDFSServer {
	friend class PVCore::PVSerializeObject;
public:
	typedef boost::shared_ptr<PVInputHDFSServer> p_type;
public:
	PVInputHDFSServer(QString const& host, uint16_t port, QString const& user);
	~PVInputHDFSServer();
protected:
	PVInputHDFSServer();

public:
	bool connect();
	void disconnect();

	inline QString const& get_host() const { return _host; }
	inline uint16_t get_port() const { return _port; }
	inline QString const& get_user() const { return _user; }

	inline hdfsFS get_hdfs() const { return _fs; }

	inline QString const& get_human_name() const { return _human_name; }

protected:
	void compute_human_name();

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

protected:
	QString _host;
	uint16_t _port;
	QString _user;
	
	hdfsFS _fs;

	QString _human_name;
};

typedef PVInputHDFSServer::p_type PVInputHDFSServer_p;

}

#endif
