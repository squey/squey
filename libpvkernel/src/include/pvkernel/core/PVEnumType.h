/**
 * \file PVEnumType.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVCORE_PVENUMTYPE_H
#define PVCORE_PVENUMTYPE_H

#include <pvkernel/core/general.h>
#include <QMetaType>
#include <QString>
#include <QStringList>
#include <assert.h>

#include <pvkernel/core/PVArgument.h>

namespace PVCore {

/**
 * \class PVEnumType
 *
 * \note This class is fully implemented in its definition, so no LibKernelDecl is needed (each library will have its own version).
 */
class PVEnumType : public PVArgumentType<PVEnumType>
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

	QString to_string() const
	{
		return QString::number(_sel) ;
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const
	{
		bool res_ok = false;

		PVArgument arg;
		arg.setValue(PVEnumType(_list, str.toInt(&res_ok)));

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	bool operator==(const PVEnumType &other) const
	{
		return _list == other._list && _sel == other._sel ;
	}

protected:
	QStringList _list;
	int _sel;
};
}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVEnumType)


#endif
