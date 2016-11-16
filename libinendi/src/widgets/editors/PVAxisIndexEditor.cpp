/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVAxisIndexType.h>

#include <inendi/PVView.h>

#include <inendi/widgets/editors/PVAxisIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVAxisIndexEditor::PVAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexEditor::PVAxisIndexEditor(Inendi::PVView const& view, QWidget* parent)
    : QComboBox(parent), _view(view)
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
	PVCombCol comb_col(currentIndex());
	PVCol index = _view.get_axes_combination().get_nraw_axis(comb_col);
	return PVCore::PVAxisIndexType(index, false, comb_col);
}
