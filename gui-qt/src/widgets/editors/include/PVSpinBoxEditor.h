//! \file PVSpinBoxEditor.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVSPINBOXEDITOR_H
#define PVCORE_PVSPINBOXEDITOR_H

#include <QSpinBox>
#include <QString>
#include <QWidget>

#include <pvcore/general.h>
#include <pvcore/PVSpinBoxType.h>

#include <picviz/PVView.h>

namespace PVInspector {

/**
 * \class PVSpinBoxEditor
 */
class PVSpinBoxEditor : public QSpinBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVSpinBoxType _s READ get_spin WRITE set_spin USER true)

public:
	PVSpinBoxEditor(Picviz::PVView& view, QWidget *parent = 0);
	virtual ~PVSpinBoxEditor();

public:
	PVCore::PVSpinBoxType get_spin() const;
	void set_spin(PVCore::PVSpinBoxType s);

protected:
	Picviz::PVView& _view;
	PVCore::PVSpinBoxType _s;
};

}

#endif // PVCORE_PVSPINBOXEDITOR_H
