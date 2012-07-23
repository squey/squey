/**
 * \file PVInputHDFSServer.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVInputHDFSServer.h"


PVRush::PVInputHDFSServer::PVInputHDFSServer(QString const& host, uint16_t port, QString const& user)
{
	_host = host;
	_port = port;
	_user = user;
	_fs = NULL;

	compute_human_name();
}

PVRush::PVInputHDFSServer::PVInputHDFSServer()
{
	_fs = NULL;
	_port = 0;
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

void PVRush::PVInputHDFSServer::compute_human_name()
{
	_human_name = QString("hdfs://%1@%2:%3/").arg(_user).arg(_host).arg(_port);
}

void PVRush::PVInputHDFSServer::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
{
	so.attribute("host", _host);
	so.attribute("port", _port);
	so.attribute("user", _user);
	compute_human_name();
}
