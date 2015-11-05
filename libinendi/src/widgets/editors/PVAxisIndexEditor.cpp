/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <inendi/PVView.h>

#include <inendi/widgets/editors/PVAxisIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVAxisIndexEditor::PVAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexEditor::PVAxisIndexEditor(Inendi::PVView const& view, QWidget *parent):
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
