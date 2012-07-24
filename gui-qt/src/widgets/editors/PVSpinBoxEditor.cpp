/**
 * \file PVSpinBoxEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSpinBoxType.h>

#include <picviz/PVView.h>

#include <PVSpinBoxEditor.h>

/******************************************************************************
 *
 * PVCore::PVSpinBoxEditor::PVSpinBoxEditor
 *
 *****************************************************************************/
PVInspector::PVSpinBoxEditor::PVSpinBoxEditor(Picviz::PVView& view, QWidget *parent):
	QSpinBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVInspector::PVSpinBoxEditor::~PVSpinBoxEditor
 *
 *****************************************************************************/
PVInspector::PVSpinBoxEditor::~PVSpinBoxEditor()
{
}

/******************************************************************************
 *
 * PVInspector::PVSpinBoxEditor::set_spin
 *
 *****************************************************************************/
void PVInspector::PVSpinBoxEditor::set_spin(PVCore::PVSpinBoxType s)
{
	setRange(1, _view.get_qtnraw_parent().get_nrows());
	setValue(s.get_value());
}

/******************************************************************************
 *
 * PVInspector::PVSpinBoxEditor::get_spin
 *
 *****************************************************************************/
PVCore::PVSpinBoxType PVInspector::PVSpinBoxEditor::get_spin() const
{
	PVCore::PVSpinBoxType ret(value());
	return ret;
}
