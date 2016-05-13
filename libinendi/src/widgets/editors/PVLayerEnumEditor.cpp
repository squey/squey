/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVView.h>

#include <inendi/widgets/editors/PVLayerEnumEditor.h>

/******************************************************************************
 *
 * PVCore::PVLayerEnumEditor::PVLayerEnumEditor
 *
 *****************************************************************************/
PVWidgets::PVLayerEnumEditor::PVLayerEnumEditor(Inendi::PVView const& view, QWidget* parent)
    : QComboBox(parent), _view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVLayerEnumEditor::~PVLayerEnumEditor
 *
 *****************************************************************************/
PVWidgets::PVLayerEnumEditor::~PVLayerEnumEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVLayerEnumEditor::set_enum
 *
 *****************************************************************************/
void PVWidgets::PVLayerEnumEditor::set_layer(Inendi::PVLayer* l)
{
	clear();
	Inendi::PVLayerStack const& ls = _view.get_layer_stack();
	int index_sel = 0;
	for (int i = 0; i < ls.get_layer_count(); i++) {
		Inendi::PVLayer* layer = (Inendi::PVLayer*)&ls.get_layer_n(i);
		addItem(layer->get_name(), QVariant::fromValue(layer));
		if (layer == l) {
			index_sel = i;
		}
	}
	setCurrentIndex(index_sel);
}

Inendi::PVLayer* PVWidgets::PVLayerEnumEditor::get_layer() const
{
	int index_sel = currentIndex();
	Inendi::PVLayer* cur_layer = (Inendi::PVLayer*)&_view.get_current_layer();
	if (index_sel == -1) {
		return cur_layer;
	}
	Inendi::PVLayer* sel_layer = itemData(index_sel).value<Inendi::PVLayer*>();
	return (_view.get_layer_stack().contains_layer(sel_layer)) ? sel_layer : cur_layer;
}
