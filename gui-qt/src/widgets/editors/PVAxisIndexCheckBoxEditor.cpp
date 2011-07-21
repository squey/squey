//! \file PVAxisIndexEditor.cpp
//! $Id: PVAxisIndexEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/general.h>
#include <pvcore/PVAxisIndexType.h>

#include <picviz/PVView.h>

#include <PVAxisIndexCheckBoxEditor.h>

/******************************************************************************
 *
 * PVCore::PVAxisIndexCheckBoxEditor::PVAxisIndexCheckBoxEditor
 *
 *****************************************************************************/
PVInspector::PVAxisIndexCheckBoxEditor::PVAxisIndexCheckBoxEditor(Picviz::PVView& view, QWidget *parent):
	QComboBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVInspector::PVAxisIndexCheckBoxEditor::~PVAxisIndexCheckBoxEditor
 *
 *****************************************************************************/
PVInspector::PVAxisIndexCheckBoxEditor::~PVAxisIndexCheckBoxEditor()
{
}

/******************************************************************************
 *
 * PVInspector::PVAxisIndexCheckBoxEditor::set_axis_index
 *
 *****************************************************************************/
void PVInspector::PVAxisIndexCheckBoxEditor::set_axis_index(PVCore::PVAxisIndexCheckBoxType axis_index)
{
	clear();
	addItems(_view.get_axes_names_list());
}

PVCore::PVAxisIndexCheckBoxType PVInspector::PVAxisIndexCheckBoxEditor::get_axis_index() const
{
	// 1 should be replace by the check to know if it is checked
	return PVCore::PVAxisIndexCheckBoxType(currentIndex(), 1);
}
