#include "PVInputHDFSServer.h"


PVRush::PVInputHDFSServer::PVInputHDFSServer(QString const& host, uint16_t port, QString const& user)
{
	_host = host;
	_port = port;
	_user = user;
	_fs = NULL;
	_human_name = QString("hdfs://%1@%2:%3/").arg(_user).arg(_host).arg(_port);
}

PVRush::PVInputHDFSServer::~PVInputHDFSServer()
{
	disconnect();
}

bool PVRush::PVInputHDFSServer::connect()
{
	if (_fs)
		return true;
	_fs = hdfsConnectAsUser(qPrintable(_host), _port, qPrintable(_user));
	if (!_fs) {
		PVLOG_INFO("Unable to connect to %s.\n", qPrintable(_human_name));
		return false;
	}
	return true;
}

void PVRush::PVInputHDFSServer::disconnect()
{
	if (_fs) {
		hdfsDisconnect(_fs);
		_fs = NULL;
	}
}
