/**
 * \file PVFormatVersion.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFORMAT_VERSION_H
#define PVFORMAT_VERSION_H

#include <pvkernel/core/general.h>
#include <QDomDocument>

namespace PVRush {

class LibKernelDecl PVFormatVersion
{
public:
	static bool to_current(QDomDocument& doc);
private:
	static bool from0to1(QDomDocument& doc);
	static bool from1to2(QDomDocument& doc);
	static bool from2to3(QDomDocument& doc);
	static bool from3to4(QDomDocument& doc);
	static bool from4to5(QDomDocument& doc);
private:
	static bool _rec_0to1(QDomElement doc);
	static bool _rec_1to2(QDomElement doc);
	static bool _rec_2to3(QDomElement doc);
	static bool _rec_3to4(QDomNode doc);
	static bool _rec_4to5(QDomNode doc);
	static QString get_version(QDomDocument const& doc);
};

}

#endif

