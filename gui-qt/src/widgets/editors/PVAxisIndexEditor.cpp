//! \file PVAxisIndexEditor.cpp
//! $Id: PVAxisIndexEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/general.h>
#include <pvcore/PVAxisIndexType.h>

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
	return PVCore::PVAxisIndexType(_view.get_real_axis_index(currentIndex()));
}
