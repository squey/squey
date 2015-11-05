/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVUnicodeString.h>
#include "PVIPv4DefaultSortingFunc.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <QtNetwork/QHostAddress>

Inendi::PVIPv4DefaultSortingFunc::PVIPv4DefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Inendi::PVIPv4DefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	return args;
}

Inendi::PVSortingFunc_f Inendi::PVIPv4DefaultSortingFunc::f()
{
	return &comp_asc;
}

Inendi::PVSortingFunc_fequals Inendi::PVIPv4DefaultSortingFunc::f_equals()
{
	return &equals_asc;
}

Inendi::PVSortingFunc_flesser Inendi::PVIPv4DefaultSortingFunc::f_lesser()
{
	return &lesser_asc;
}

Inendi::PVQtSortingFunc_f Inendi::PVIPv4DefaultSortingFunc::qt_f()
{
	return &qt_comp_asc;
}

Inendi::PVQtSortingFunc_fequals Inendi::PVIPv4DefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_asc;
}

Inendi::PVQtSortingFunc_flesser Inendi::PVIPv4DefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_asc;
}

bool Inendi::PVIPv4DefaultSortingFunc::lesser_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	quint32 f1 = QHostAddress(s1.get_qstr(s)).toIPv4Address();
	quint32 f2 = QHostAddress(s2.get_qstr(s)).toIPv4Address();
	return f1 < f2;
}

bool Inendi::PVIPv4DefaultSortingFunc::equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

int Inendi::PVIPv4DefaultSortingFunc::comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	quint32 f1 = QHostAddress(s1.get_qstr(s)).toIPv4Address();
	quint32 f2 = QHostAddress(s2.get_qstr(s)).toIPv4Address();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}

bool Inendi::PVIPv4DefaultSortingFunc::qt_lesser_asc(QString const& s1, QString const& s2)
{
	quint32 f1 = QHostAddress(s1).toIPv4Address();
	quint32 f2 = QHostAddress(s2).toIPv4Address();
	return f1 < f2;
}

bool Inendi::PVIPv4DefaultSortingFunc::qt_equals_asc(QString const& s1, QString const& s2)
{
	quint32 f1 = QHostAddress(s1).toIPv4Address();
	quint32 f2 = QHostAddress(s2).toIPv4Address();
	return f1 == f2;
}

int Inendi::PVIPv4DefaultSortingFunc::qt_comp_asc(QString const& s1, QString const& s2)
{
	quint32 f1 = QHostAddress(s1).toIPv4Address();
	quint32 f2 = QHostAddress(s2).toIPv4Address();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
