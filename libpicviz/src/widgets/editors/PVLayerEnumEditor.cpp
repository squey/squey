#include <pvkernel/core/general.h>
#include <picviz/PVView.h>

#include <picviz/widgets/editors/PVLayerEnumEditor.h>

/******************************************************************************
 *
 * PVCore::PVLayerEnumEditor::PVLayerEnumEditor
 *
 *****************************************************************************/
PVWidgets::PVLayerEnumEditor::PVLayerEnumEditor(Picviz::PVView const& view, QWidget *parent):
	QComboBox(parent),
	_view(view)
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
void PVWidgets::PVLayerEnumEditor::set_layer(Picviz::PVLayer* l)
{
	clear();
	Picviz::PVLayerStack const& ls = _view.get_layer_stack();
	int index_sel = 0;
	for (int i = 0; i < ls.get_layer_count(); i++) {
		Picviz::PVLayer* layer = (Picviz::PVLayer*) &ls.get_layer_n(i);
		addItem(layer->get_name(), QVariant::fromValue(layer));
		if (layer == l) {
			index_sel = i;
		}
	}
	setCurrentIndex(index_sel);
}

Picviz::PVLayer* PVWidgets::PVLayerEnumEditor::get_layer() const
{
	int index_sel = currentIndex();
	Picviz::PVLayer* cur_layer = (Picviz::PVLayer*) &_view.get_current_layer();
	if (index_sel == -1) {
		return cur_layer;
	}
	Picviz::PVLayer* sel_layer = itemData(index_sel).value<Picviz::PVLayer*>();
	return (_view.get_layer_stack().contains_layer(sel_layer)) ? sel_layer : cur_layer;
}
