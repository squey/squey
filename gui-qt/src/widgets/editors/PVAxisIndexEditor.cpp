/**
 * \file PVAxisIndexEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <picviz/PVView.h>

#include <PVAxisIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVAxisIndexEditor::PVAxisIndexEditor
 *
 *****************************************************************************/
PVInspector::PVAxisIndexEditor::PVAxisIndexEditor(Picviz::PVView& view, QWidget *parent):
	QComboBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVInspector::PVAxisIndexEditor::~PVAxisIndexEditor
 *
 *****************************************************************************/
PVInspector::PVAxisIndexEditor::~PVAxisIndexEditor()
{
}

/******************************************************************************
 *
 * PVInspector::PVAxisIndexEditor::set_axis_index
 *
 *****************************************************************************/
void PVInspector::PVAxisIndexEditor::set_axis_index(PVCore::PVAxisIndexType axis_index)
{
	clear();
	addItems(_view.get_axes_names_list());
	setCurrentIndex(_view.axes_combination.get_combined_axis_column_index(axis_index.get_original_index()));
}

PVCore::PVAxisIndexType PVInspector::PVAxisIndexEditor::get_axis_index() const
{
	int index = _view.axes_combination.get_axis_column_index(currentIndex());
	return PVCore::PVAxisIndexType(index);
}
