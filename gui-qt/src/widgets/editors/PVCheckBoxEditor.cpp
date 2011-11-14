//! \file PVCheckBoxEditor.cpp
//! $Id: PVCheckBoxEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVCheckBoxType.h>

#include <picviz/PVView.h>

#include <PVCheckBoxEditor.h>

/******************************************************************************
 *
 * PVInspector::PVCheckBoxEditor::PVCheckBoxEditor
 *
 *****************************************************************************/
PVInspector::PVCheckBoxEditor::PVCheckBoxEditor(Picviz::PVView& view, QWidget *parent):
	QCheckBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVInspector::PVCheckBoxEditor::~PVCheckBoxEditor
 *
 *****************************************************************************/
PVInspector::PVCheckBoxEditor::~PVCheckBoxEditor()
{
}

/******************************************************************************
 *
 * PVInspector::PVCheckBoxEditor::set_checked
 *
 *****************************************************************************/
void PVInspector::PVCheckBoxEditor::set_checked(PVCore::PVCheckBoxType e)
{
	_e = e;
	if (e.get_checked()) {
		setCheckState(Qt::Checked);
	} else {
		setCheckState(Qt::Unchecked);
	}
}

PVCore::PVCheckBoxType PVInspector::PVCheckBoxEditor::get_checked() const
{
	PVCore::PVCheckBoxType ret(_e);
	ret.set_checked(isChecked());
	return ret;
}
