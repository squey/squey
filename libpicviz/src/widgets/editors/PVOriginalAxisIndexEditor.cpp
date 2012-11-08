/**
 * \file PVOriginalAxisIndexEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <picviz/PVView.h>

#include <picviz/widgets/editors/PVOriginalAxisIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVOriginalAxisIndexEditor::PVOriginalAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVOriginalAxisIndexEditor::PVOriginalAxisIndexEditor(Picviz::PVView const& view, QWidget *parent):
	QComboBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVOriginalAxisIndexEditor::~PVOriginalAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVOriginalAxisIndexEditor::~PVOriginalAxisIndexEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVOriginalAxisIndexEditor::set_axis_index
 *
 *****************************************************************************/
void PVWidgets::PVOriginalAxisIndexEditor::set_axis_index(PVCore::PVOriginalAxisIndexType axis_index)
{
	clear();
	addItems(_view.get_original_axes_names_list());
	setCurrentIndex(axis_index.get_original_index());
}

PVCore::PVOriginalAxisIndexType PVWidgets::PVOriginalAxisIndexEditor::get_axis_index() const
{
	int index = currentIndex();
	return PVCore::PVOriginalAxisIndexType(index, false);
}
