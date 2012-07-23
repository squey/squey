/**
 * \file PVTextEditType.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVTEXTEDITTYPE_H
#define PVCORE_PVTEXTEDITTYPE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>

#include <QString>
#include <QMetaType>

namespace PVCore {

class PVTextEditType
{
public:
	inline void set_text(QString const& txt) { _txt = txt; }
	inline QString const& get_text() const { return _txt; }
private:
	QString _txt;
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVTextEditType)

#endif
