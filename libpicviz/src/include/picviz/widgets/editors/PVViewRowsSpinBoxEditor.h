/**
 * \file PVViewRowsSpinBoxEditor.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVSPINBOXEDITOR_H
#define PVCORE_PVSPINBOXEDITOR_H

#include <QSpinBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSpinBoxType.h>

namespace Picviz {
class PVView;
}

namespace PVWidgets {

class PVViewRowsSpinBoxEditor : public QSpinBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVSpinBoxType _s READ get_spin WRITE set_spin USER true)

public:
	PVViewRowsSpinBoxEditor(Picviz::PVView const& view, QWidget *parent = 0);
	virtual ~PVViewRowsSpinBoxEditor();

public:
	PVCore::PVSpinBoxType get_spin() const;
	void set_spin(PVCore::PVSpinBoxType s);

protected:
	PVCore::PVSpinBoxType _s;
	Picviz::PVView const& _view;
};

}

#endif // PVCORE_PVSPINBOXEDITOR_H
