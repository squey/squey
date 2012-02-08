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

class LibKernelDecl PVArgumentKey: public QString
{
public:
	PVArgumentKey(QString const& key, QString const& desc = QString()):
		QString(key),
		_desc(desc)
	{
		if (desc.isNull()) {
			set_desc_from_key();
		}
		else {
			_key_desc[key] = desc;
		}
	}
	PVArgumentKey(const char* key):
		QString(key)
	{
		set_desc_from_key();
	}

	QString const& key() const { return *((QString*)this); }
	QString const& desc() const { return _desc; }

private:
	void set_desc_from_key()
	{
		_desc = _key_desc.value(*this, *this);
	}
private:
	QString _desc;
	static QHash<QString, QString> _key_desc;
};

}

extern unsigned int LibKernelDecl qHash(PVCore::PVArgumentKey const& key);

namespace PVCore {

typedef QVariant                           PVArgument;
typedef QHash<PVArgumentKey,PVArgument>    PVArgumentList;

LibKernelDecl QString PVArgument_to_QString(PVArgument const& v);
LibKernelDecl PVArgument QString_to_PVArgument(QString const& v);

LibKernelDecl void dump_argument_list(PVArgumentList const& l);

void PVArgumentList_set_common_args_from(PVCore::PVArgumentList& ret, PVCore::PVArgumentList const& ref);

}


#endif
