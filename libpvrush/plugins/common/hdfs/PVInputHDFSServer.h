#ifndef PVINPUTHDFSSERVER_FILE_H
#define PVINPUTHDFSSERVER_FILE_H

#include <pvcore/general.h>
#include <pvcore/stdint.h>

#include <boost/shared_ptr.hpp>

#include <hadoop/hdfs.h>


namespace PVRush {

class LibExport PVInputHDFSServer {
public:
	typedef boost::shared_ptr<PVInputHDFSServer> p_type;
public:
	PVInputHDFSServer(QString const& host, uint16_t port, QString const& user);
	~PVInputHDFSServer();
public:
	bool connect();
	void disconnect();

	inline QString const& get_host() const { return _host; }
	inline uint16_t get_port() const { return _port; }
	inline QString const& get_user() const { return _user; }

	inline hdfsFS get_hdfs() const { return _fs; }

	inline QString const& get_human_name() const { return _human_name; }
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
