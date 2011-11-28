//! \file PVArgument.h
//! $Id: PVArgument.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVARGUMENT_H
#define PVCORE_PVARGUMENT_H

#include <pvkernel/core/general.h>
#include <QHash>
#include <QString>
#include <QVariant>

namespace PVCore {

class PVArgumentKey: public QString
{
public:
	PVArgumentKey(QString const& key, QString const& desc = QString()):
		QString(key),
		_desc(desc)
	{
		if (desc.isNull()) {
			_desc = *this;
		}
	}
	PVArgumentKey(const char* key):
		QString(key),
		_desc(key)
	{ }

	QString const& key() const { return *((QString*)this); }
	QString const& desc() const { return _desc; }

private:
	QString _desc;
};

}

unsigned int qHash(PVCore::PVArgumentKey const& key);

namespace PVCore {

typedef QVariant                           PVArgument;
typedef QHash<PVArgumentKey,PVArgument>    PVArgumentList;

LibKernelDecl QString PVArgument_to_QString(PVArgument const& v);
LibKernelDecl PVArgument QString_to_PVArgument(QString const& v);

LibKernelDecl void dump_argument_list(PVArgumentList const& l);

}


#endif
