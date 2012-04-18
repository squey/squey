//! \file PVViewRowsSpinBoxEditor.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSpinBoxType.h>

#include <picviz/PVView.h>
#include <picviz/widgets/editors/PVViewRowsSpinBoxEditor.h>

/******************************************************************************
 *
 * PVCore::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor(Picviz::PVView& view, QWidget *parent):
	QSpinBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVViewRowsSpinBoxEditor::~PVViewRowsSpinBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVViewRowsSpinBoxEditor::~PVViewRowsSpinBoxEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVViewRowsSpinBoxEditor::set_spin
 *
 *****************************************************************************/
void PVWidgets::PVViewRowsSpinBoxEditor::set_spin(PVCore::PVSpinBoxType s)
{
	setRange(1, _view.get_qtnraw_parent().get_nrows());
	setValue(s.get_value());
}

/******************************************************************************
 *
 * PVWidgets::PVViewRowsSpinBoxEditor::get_spin
 *
 *****************************************************************************/
PVCore::PVSpinBoxType PVWidgets::PVViewRowsSpinBoxEditor::get_spin() const
{
	PVCore::PVSpinBoxType ret(value());
	return ret;
}
