/**
 * \file PVViewRowsSpinBoxEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSpinBoxType.h>

#include <picviz/PVView.h>
#include <picviz/widgets/editors/PVViewRowsSpinBoxEditor.h>

/******************************************************************************
 *
 * PVCore::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor(Picviz::PVView const& view, QWidget *parent):
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
	setRange(1, _view.get_row_count());
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
