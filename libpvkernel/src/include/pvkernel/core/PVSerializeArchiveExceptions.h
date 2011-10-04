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
	QString what() { return _msg; }
protected:
	QString _msg;
};

}

#endif
