//! \file PVEnumType.h
//! $Id: PVEnumType.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVENUMTYPE_H
#define PVCORE_PVENUMTYPE_H

#include <pvkernel/core/general.h>
#include <QMetaType>
#include <QString>
#include <QStringList>
#include <assert.h>

namespace PVCore {

/**
 * \class PVEnumType
 *
 * \note This class is fully implemented in its definition, so no LibKernelDecl is needed (each library will have its own version).
 */
class PVEnumType
{
	
public:
	/**
	 * Constructor
	 */
	PVEnumType() { _sel = -1; };
	PVEnumType(QStringList const& list, int sel) { _list = list; _sel = sel; };

	QStringList const& get_list() const { return _list; }
	QString get_sel() const { assert(_sel != -1); return _list[_sel]; }
	int get_sel_index() const { return _sel; }
	void set_sel(int index) { assert(index < _list.count() && index >= 0); _sel = index; };
	bool set_sel_from_str(QString const& s) { int r = _list.indexOf(s); if (r == -1) return false; _sel = r; return true; }

protected:
	QStringList _list;
	int _sel;
};
}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVEnumType)


#endif
