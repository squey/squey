/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVAxisIndexType.h>

#include <inendi/PVView.h>

#include <pvguiqt/PVAxisIndexFilteredEditor.h>

/******************************************************************************
 *
 * PVCore::PVAxisIndexFilteredEditor::PVAxisIndexFilteredEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexFilteredEditor::PVAxisIndexFilteredEditor(
    Inendi::PVView const& view, PVDisplays::PVDisplayViewDataIf const& display_if, QWidget* parent)
    : QComboBox(parent), _view(view), _display_if(display_if)
{
}

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexFilteredEditor::set_axis_index
 *
 *****************************************************************************/
void PVWidgets::PVAxisIndexFilteredEditor::set_axis_index(PVCore::PVAxisIndexType axis_index)
{
	clear();
	auto axes_names = _view.get_axes_names_list();
	for (int i = 0; i < axes_names.size(); ++i) {
		if (_display_if.should_add_to_menu((Inendi::PVView*)&_view, {PVCombCol(i)})) {
			addItem(axes_names[i]);
			setItemData(count() - 1, QVariant(i));
		}
	}
	setCurrentIndex(axis_index.get_axis_index());
}

PVCore::PVAxisIndexType PVWidgets::PVAxisIndexFilteredEditor::get_axis_index() const
{
	PVCombCol comb_col(itemData(currentIndex()).toInt());
	PVCol index = _view.get_axes_combination().get_nraw_axis(comb_col);
	return PVCore::PVAxisIndexType(index, false, comb_col);
}
