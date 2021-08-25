//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
