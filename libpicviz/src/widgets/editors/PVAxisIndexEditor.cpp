//! \file PVAxisIndexEditor.cpp
//! $Id: PVAxisIndexEditor.cpp 2699 2011-05-12 03:58:48Z cdash $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <picviz/PVView.h>

#include <picviz/widgets/editors/PVAxisIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVAxisIndexEditor::PVAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexEditor::PVAxisIndexEditor(Picviz::PVView const& view, QWidget *parent):
	QComboBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexEditor::~PVAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexEditor::~PVAxisIndexEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexEditor::set_axis_index
 *
 *****************************************************************************/
void PVWidgets::PVAxisIndexEditor::set_axis_index(PVCore::PVAxisIndexType axis_index)
{
	clear();
	addItems(_view.get_axes_names_list());
	setCurrentIndex(_view.axes_combination.get_combined_axis_column_index(axis_index.get_original_index()));
}

PVCore::PVAxisIndexType PVWidgets::PVAxisIndexEditor::get_axis_index() const
{
	int index = _view.axes_combination.get_axis_column_index(currentIndex());
	return PVCore::PVAxisIndexType(index);
}
