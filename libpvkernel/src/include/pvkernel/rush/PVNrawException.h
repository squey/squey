#ifndef PVRUSH_PVNRAWEXCEPTION_H
#define PVRUSH_PVNRAWEXCEPTION_H

#include <exception>

namespace PVRush {

class PVNrawException: public std::exception
{
public:
	PVNrawException(QString const& str):
		_msg(str.toLocal8Bit())
	{ }

	~PVNrawException() throw()
	{ }

public:
	const char* what() const throw() override { return _msg.constData(); }

private:
	QByteArray _msg;
};

}

#endif
