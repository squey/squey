/**
 * \file PVAxisIndexEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
	setCurrentIndex(axis_index.get_axis_index());
}

PVCore::PVAxisIndexType PVWidgets::PVAxisIndexEditor::get_axis_index() const
{
	int index = _view.axes_combination.get_axis_column_index(currentIndex());
	return PVCore::PVAxisIndexType(index, false, currentIndex());
}
