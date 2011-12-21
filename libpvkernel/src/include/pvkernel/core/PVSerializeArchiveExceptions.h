#ifndef PVCORE_PVSERIALIZEARCHIVEEXCEPTIONS_H
#define PVCORE_PVSERIALIZEARCHIVEEXCEPTIONS_H

#include <pvkernel/core/general.h>
#include <QString>


namespace PVCore {

namespace priv {

class LibKernelDecl PVSerializeArchiveErrorBase
{
public:
	virtual ~PVSerializeArchiveErrorBase() { }
	virtual QString const& what() const = 0;
};

}

class LibKernelDecl PVSerializeArchiveError: public priv::PVSerializeArchiveErrorBase
{
public:
	PVSerializeArchiveError(QString const& msg):
		_msg(msg)
	{ }

public:
	QString const& what() const { return _msg; }
protected:
	QString _msg;
};

class LibKernelDecl PVSerializeArchiveErrorNoObject: public PVSerializeArchiveError
{
public:
	PVSerializeArchiveErrorNoObject(QString const& obj, QString const& msg):
		PVSerializeArchiveError(msg),
		_obj(obj)
	{ }
public:
	QString const& obj() const { return _obj; }
protected:
	QString _obj;
};

class LibKernelDecl PVSerializeArchiveErrorFileNotReadable: public PVSerializeArchiveError
{
public:
	PVSerializeArchiveErrorFileNotReadable(QString const& path):
		PVSerializeArchiveError(QString("File %1 does not exist or is not readable.").arg(path)),
		_path(path)
	{ }
public:
	QString const& get_path() const { return _path; }
protected:
	QString _path;
};

}

#endif
