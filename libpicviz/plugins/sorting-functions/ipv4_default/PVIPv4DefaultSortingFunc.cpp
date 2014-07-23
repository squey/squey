/**
 * \file PVIPv4DefaultSortingFunc.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString.h>
#include "PVIPv4DefaultSortingFunc.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <QtNetwork/QHostAddress>

static inline in_addr_t ipv4_str2ul(const PVCore::PVUnicodeString::utf_char* text)
{
	return ntohl(inet_addr((char*)text));
}

Picviz::PVIPv4DefaultSortingFunc::PVIPv4DefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Picviz::PVIPv4DefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	return args;
}

Picviz::PVSortingFunc_f Picviz::PVIPv4DefaultSortingFunc::f()
{
	return &comp_asc;
}

Picviz::PVSortingFunc_fequals Picviz::PVIPv4DefaultSortingFunc::f_equals()
{
	return &equals_asc;
}

Picviz::PVSortingFunc_flesser Picviz::PVIPv4DefaultSortingFunc::f_lesser()
{
	return &lesser_asc;
}

Picviz::PVQtSortingFunc_f Picviz::PVIPv4DefaultSortingFunc::qt_f()
{
	return &qt_comp_asc;
}

Picviz::PVQtSortingFunc_fequals Picviz::PVIPv4DefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_asc;
}

Picviz::PVQtSortingFunc_flesser Picviz::PVIPv4DefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_asc;
}

bool Picviz::PVIPv4DefaultSortingFunc::lesser_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	in_addr_t f1 = ipv4_str2ul(s1.buffer());
	in_addr_t f2 = ipv4_str2ul(s2.buffer());
	return f1 < f2;
}

bool Picviz::PVIPv4DefaultSortingFunc::equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

int Picviz::PVIPv4DefaultSortingFunc::comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	in_addr_t f1 = ipv4_str2ul(s1.buffer());
	in_addr_t f2 = ipv4_str2ul(s2.buffer());
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}

bool Picviz::PVIPv4DefaultSortingFunc::qt_lesser_asc(QString const& s1, QString const& s2)
{
	quint32 f1 = QHostAddress(s1).toIPv4Address();
	quint32 f2 = QHostAddress(s2).toIPv4Address();
	return f1 < f2;
}

bool Picviz::PVIPv4DefaultSortingFunc::qt_equals_asc(QString const& s1, QString const& s2)
{
	quint32 f1 = QHostAddress(s1).toIPv4Address();
	quint32 f2 = QHostAddress(s2).toIPv4Address();
	return f1 == f2;
}

int Picviz::PVIPv4DefaultSortingFunc::qt_comp_asc(QString const& s1, QString const& s2)
{
	quint32 f1 = QHostAddress(s1).toIPv4Address();
	quint32 f2 = QHostAddress(s2).toIPv4Address();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
