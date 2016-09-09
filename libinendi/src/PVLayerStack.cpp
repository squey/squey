/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLayer.h>
#include <inendi/PVLayerStack.h>
#include <inendi/PVSelection.h>

#include <QApplication>
#include <QClipboard>

/******************************************************************************
 *
 * Inendi::PVLayerStack::PVLayerStack
 *
 *****************************************************************************/
Inendi::PVLayerStack::PVLayerStack()
{
	_selected_layer_index = -1;
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::get_new_layer_name
 *
 *****************************************************************************/
QString Inendi::PVLayerStack::get_new_layer_name() const
{
	return QString("New layer %1").arg(get_layer_count() + 1);
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::hide_layers
 *
 *****************************************************************************/
void Inendi::PVLayerStack::hide_layers()
{
	for (int i = 0; i < get_layer_count(); i++) {
		PVLayer& layer = get_layer_n(i);
		layer.set_visible(false);
	}
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::append_layer
 *
 *****************************************************************************/
Inendi::PVLayer* Inendi::PVLayerStack::append_new_layer(PVRow row_count,
                                                        QString const& name /*= QString()*/)
{
	/* We prepare the string for the name of the new layer */
	QString layer_name;
	if (name.isEmpty()) {
		layer_name = get_new_layer_name();
	} else {
		layer_name = name;
	}
	return append_layer(PVLayer(layer_name, row_count));
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::append_layer
 *
 *****************************************************************************/
Inendi::PVLayer* Inendi::PVLayerStack::append_layer(const PVLayer& layer)
{
	/* We test if we have not reached the maximal number of layers */
	if (get_layer_count() < INENDI_LAYER_STACK_MAX_DEPTH - 1) {
		_table.append(layer);
		_selected_layer_index = get_layer_count() - 1;

		return &_table.last();
	}
	// FIXME: should have an exception here, that will be treated by the GUI !
	return nullptr;
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::append_new_layer_from_selection_and_lines_properties
 *
 *****************************************************************************/
Inendi::PVLayer* Inendi::PVLayerStack::append_new_layer_from_selection_and_lines_properties(
    PVSelection const& selection, PVLinesProperties const& lines_properties)
{
	/* We prepare the string for the name of the new layer */
	QString new_layer_automatic_name = QString("New layer %1").arg(get_layer_count());
	return append_layer(PVLayer(new_layer_automatic_name, selection, lines_properties));
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::delete_by_index
 *
 *****************************************************************************/
void Inendi::PVLayerStack::delete_by_index(int index)
{
	if ((index < 0) || (index >= get_layer_count())) {
		return;
	}

	if (_table.at(index).is_locked()) {
		return;
	}

	_table.removeAt(index);

	// we set the selected layer according to the posibilities
	if (_selected_layer_index == get_layer_count()) {
		_selected_layer_index = get_layer_count() - 1;
	}

	// and we make sure it is visible
	_table[_selected_layer_index].set_visible(true);
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::delete_selected_layer
 *
 *****************************************************************************/
void Inendi::PVLayerStack::delete_selected_layer()
{
	delete_by_index(_selected_layer_index);
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::duplicate_selected_layer
 *
 *****************************************************************************/
Inendi::PVLayer* Inendi::PVLayerStack::duplicate_selected_layer(const QString& name)
{
	if ((_selected_layer_index < 0) || (_selected_layer_index >= get_layer_count())) {
		return nullptr;
	}

	const PVLayer& selected_layer = _table.at(_selected_layer_index);
	PVLayer* new_layer = append_new_layer(selected_layer.get_selection().count(), name);

	new_layer->get_selection() = selected_layer.get_selection();
	new_layer->get_lines_properties() = selected_layer.get_lines_properties();

	return new_layer;
}

/******************************************************************************
 *
 * Inendi::PVLayerStack::delete_all_layers
 *
 *****************************************************************************/
void Inendi::PVLayerStack::delete_all_layers()
{
	_table.clear();
	_selected_layer_index = -1;
}

/**********************************************************************
*
* Inendi::PVLayerStack::move_layer_down
*
**********************************************************************/
void Inendi::PVLayerStack::move_layer_down(int index)
{
	if ((0 < index) && (index < get_layer_count())) {
		_table.move(index, index - 1);
	}
}

/**********************************************************************
*
* Inendi::PVLayerStack::move_layer_up
*
**********************************************************************/
void Inendi::PVLayerStack::move_layer_up(int index)
{
	if (index < (get_layer_count() - 1)) {
		_table.move(index, index + 1);
	}
}

/**********************************************************************
* Inendi::PVLayerStack::move_selected_layer_to
**********************************************************************/

void Inendi::PVLayerStack::move_selected_layer_to(int new_index)
{
	if ((new_index < 0) || (new_index >= get_layer_count())) {
		return;
	}

	_table.move(_selected_layer_index, new_index);
	_selected_layer_index = new_index;
}

/**********************************************************************
*
* Inendi::PVLayerStack::move_selected_layer_down
*
**********************************************************************/
void Inendi::PVLayerStack::move_selected_layer_down()
{
	if (_selected_layer_index > 0) {
		move_layer_down(_selected_layer_index);
		_selected_layer_index -= 1;
	}
}

/**********************************************************************
 *
 * Inendi::PVLayerStack::move_selected_layer_up
 *
 **********************************************************************/
void Inendi::PVLayerStack::move_selected_layer_up()
{
	if (_selected_layer_index < get_layer_count() - 1) {
		move_layer_up(_selected_layer_index);
		_selected_layer_index += 1;
	}
}

/**********************************************************************
 *
 * Inendi::PVLayerStack::process
 *
 **********************************************************************/
void Inendi::PVLayerStack::process(PVLayer& output_layer, PVRow row_count)
{
	/******************************
	* Preparation
	******************************/
	/* We prepare a pointer to the layer of the LS being processed */
	PVLayer* layer_being_processed;
	/* We prepare a temporary selection, needed in our computations */
	PVSelection temp_selection(row_count);
	/* We store locally the layer-stack->layer_count and prepare a counter */
	int i;

	/******************************
	* Initializations
	******************************/
	/* It's time to erase the output_layer ! */
	output_layer.reset_to_empty_and_default_color();

	/******************************
	* Main computation
	******************************/
	/* The layer-stack might be empty so we have to be carefull... */
	if (get_layer_count() > 0) {
		/* We process the layers from TOP to BOTTOM, that is, from
		*  the most visible to the less visible */
		for (i = get_layer_count() - 1; i >= 0; i--) {
			/* We prepare a direct access to the layer we have to process */
			layer_being_processed = &(_table[i]);
			/* We check if this layer is visible */
			if (layer_being_processed->get_visible()) {
				/* we compute the selection of lines present
				*  in the layer being processed but not yet
				*  in the output layer */

				// The line below is an optimized version of:
				// temp_selection = layer_being_processed->get_selection() -
				// output_layer.get_selection();
				temp_selection.AB_sub(layer_being_processed->get_selection(),
				                      output_layer.get_selection());

				/* and we already update the selection in
				*  the output_layer */
				output_layer.get_selection() |= layer_being_processed->get_selection();
				// output_layer.get_selection() |=
				// layer_being_processed->get_selection();

				/* We copy in the output_layer the only new
				*  lines properties */
				layer_being_processed->get_lines_properties().A2B_copy_restricted_by_selection(
				    output_layer.get_lines_properties(), temp_selection);
			}
		}
	}
}

void Inendi::PVLayerStack::compute_selectable_count()
{
	for (int i = 0; i < get_layer_count(); i++) {
		_table[i].compute_selectable_count();
	}
}

bool Inendi::PVLayerStack::contains_layer(PVLayer* layer) const
{
	for (PVLayer const& l : _table) {
		if (&l == layer) {
			return true;
		}
	}
	return false;
}

void Inendi::PVLayerStack::serialize_write(PVCore::PVSerializeObject& so)
{
	so.attribute("selected_layer_index", _selected_layer_index);

	PVCore::PVSerializeObject_p list_obj = so.create_object("layers", "", false, false);
	int idx = 0;
	for (PVLayer& layer : _table) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(child_name, "", false);
		layer.serialize_write(*new_obj);
		new_obj->set_bound_obj(layer);
	}
	so.attribute("layer_count", idx);
}

Inendi::PVLayerStack Inendi::PVLayerStack::serialize_read(PVCore::PVSerializeObject& so)
{

	Inendi::PVLayerStack ls;

	so.attribute("selected_layer_index", ls._selected_layer_index);

	ls._table.clear(); // Remove default layer created on view creation.

	PVCore::PVSerializeObject_p list_obj = so.create_object("layers", "", false, false);
	int layer_count;
	so.attribute("layer_count", layer_count);
	for (int idx = 0; idx < layer_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		ls._table.append(PVLayer::serialize_read(*new_obj));
	}

	return ls;
}

void Inendi::PVLayerStack::copy_details_to_clipboard()
{
	QString s;

	for (int i = get_layer_count() - 1; i >= 0; --i) {
		const auto& l = get_layer_n(i);
		s += l.get_name();
		s += "\t";
		s += QString::number(l.get_selectable_count());
		s += "\n";
	}

	QApplication::clipboard()->setText(s);
}
