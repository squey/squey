/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <inendi/PVView.h>
#include <inendi/PVSource.h>

#include <inendi/widgets/editors/PVOriginalAxisIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVOriginalAxisIndexEditor::PVOriginalAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVOriginalAxisIndexEditor::PVOriginalAxisIndexEditor(Inendi::PVView const& view,
                                                                QWidget* parent)
    : QComboBox(parent), _view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVOriginalAxisIndexEditor::set_axis_index
 *
 *****************************************************************************/
void PVWidgets::PVOriginalAxisIndexEditor::set_axis_index(
    PVCore::PVOriginalAxisIndexType axis_index)
{
	clear();
	addItems(_view.get_axes_combination().get_nraw_names());
	setCurrentIndex(axis_index.get_original_index());
}

PVCore::PVOriginalAxisIndexType PVWidgets::PVOriginalAxisIndexEditor::get_axis_index() const
{
	return PVCore::PVOriginalAxisIndexType(PVCol(currentIndex()), false);
}
