#ifndef PVCORE_PVSERIALIZEARCHIVEEXCEPTIONS_H
#define PVCORE_PVSERIALIZEARCHIVEEXCEPTIONS_H

#include <pvkernel/core/general.h>
#include <QString>

namespace PVCore {

class LibKernelDecl PVSerializeArchiveError
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

}

#endif
