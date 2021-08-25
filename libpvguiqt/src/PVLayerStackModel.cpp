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

#include <inendi/PVLayerStack.h>
#include <inendi/PVView.h>

#include <pvguiqt/PVCustomQtRoles.h>
#include <pvguiqt/PVLayerStackModel.h>

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::PVLayerStackModel
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackModel::PVLayerStackModel(Inendi::PVView& lib_view, QObject* parent)
    : QAbstractTableModel(parent)
    , _lib_view(lib_view)
    , select_brush(QColor(255, 240, 200))
    , unselect_brush(QColor(180, 180, 180))
{
	PVLOG_DEBUG("PVGuiQt::PVLayerStackModel::%s : Creating object\n", __FUNCTION__);

	select_font.setBold(true);

	lib_view._layer_stack_about_to_refresh.connect(
	    sigc::mem_fun(this, &PVGuiQt::PVLayerStackModel::layer_stack_about_to_be_refreshed));
	lib_view._layer_stack_refreshed.connect(
	    sigc::mem_fun(this, &PVGuiQt::PVLayerStackModel::layer_stack_refreshed));
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::columnCount
 *
 *****************************************************************************/
int PVGuiQt::PVLayerStackModel::columnCount(const QModelIndex& /*index*/) const
{
	return 3;
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::data
 *
 *****************************************************************************/
QVariant PVGuiQt::PVLayerStackModel::data(const QModelIndex& index, int role) const
{
	// AG: the two following lines are kept for the sake of history...
	/* We prepare a direct acces to the total number of layers */
	// int layer_count = lib_layer_stack().get_layer_count();

	// AG: this comment is also kept for history :)
	/* We create and store the true index of the layer in the lib */
	int lib_index = lib_index_from_model_index(index.row());

	switch (role) {
	case Qt::DecorationRole:
		switch (index.column()) {
		case 0:
			if (lib_layer_stack().get_layer_n(lib_index).get_visible()) {
				return QPixmap(":/layer-active.png");
			} else {
				return QPixmap(":/layer-inactive.png");
			}
			break;
		}
		break;

	case (Qt::BackgroundRole):
		if (lib_layer_stack().get_selected_layer_index() == lib_index) {
			return QBrush(QColor(205, 139, 204));
		}
		break;

	case (Qt::DisplayRole):
		switch (index.column()) {
		case 1:
			return lib_layer_stack().get_layer_n(lib_index).get_name();
		case 2:
			return QString("%L3").arg(
			    lib_layer_stack().get_layer_n(lib_index).get_selectable_count());
		}
		break;

	case (Qt::EditRole):
		switch (index.column()) {
		case 1:
			return lib_layer_stack().get_layer_n(lib_index).get_name();
		}
		break;

	case (Qt::TextAlignmentRole):
		switch (index.column()) {
		case 0:
			return (Qt::AlignCenter + Qt::AlignVCenter);

		case 2:
			return (Qt::AlignRight + Qt::AlignVCenter);

		default:
			return (Qt::AlignLeft + Qt::AlignVCenter);
		}
		break;

	case (Qt::ToolTipRole):
		switch (index.column()) {
		case 1:
			return lib_layer_stack().get_layer_n(lib_index).get_name();
			break;
		}
		break;

	case (PVCustomQtRoles::UnderlyingObject): {
		QVariant ret;
		ret.setValue<void*>((void*)&lib_layer_stack().get_layer_n(lib_index));
		return ret;
	}
	}
	return QVariant();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVGuiQt::PVLayerStackModel::flags(const QModelIndex& index) const
{
	switch (index.column()) {
	case 0:
		// return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
		return Qt::ItemIsEditable | Qt::ItemIsEnabled;

	default:
		return (Qt::ItemIsEditable | Qt::ItemIsEnabled);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::headerData
 *
 *****************************************************************************/
QVariant PVGuiQt::PVLayerStackModel::headerData(int /*section*/,
                                                Qt::Orientation /*orientation*/,
                                                int role) const
{
	// FIXME : this should not be used : delegate...
	switch (role) {
	case (Qt::SizeHintRole):
		return QSize(37, 37);
		break;
	}

	return QVariant();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::rowCount
 *
 *****************************************************************************/
int PVGuiQt::PVLayerStackModel::rowCount(const QModelIndex& /*index*/) const
{
	return lib_layer_stack().get_layer_count();
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackModel::setData
 *
 *****************************************************************************/
bool PVGuiQt::PVLayerStackModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	int layer_count = lib_layer_stack().get_layer_count();
	/* We create and store the true index of the layer in the lib */
	int lib_index = layer_count - 1 - index.row();

	switch (role) {
	case (Qt::EditRole):
		switch (index.column()) {
		case 0:
			lib_view().toggle_layer_stack_layer_n_visible_state(lib_index);
			lib_view().process_layer_stack();
			return true;

		case 1:
			lib_view().set_layer_stack_layer_n_name(lib_index, value.toString());
			return true;

		default:
			return QAbstractTableModel::setData(index, value, role);
		}

	default:
		return QAbstractTableModel::setData(index, value, role);
	}

	return false;
}

void PVGuiQt::PVLayerStackModel::layer_stack_about_to_be_refreshed()
{
	beginResetModel();
}

void PVGuiQt::PVLayerStackModel::reset_layer_colors(const int idx)
{
	Inendi::PVLayerStack& layerstack = lib_layer_stack();
	Inendi::PVLayer& layer = layerstack.get_layer_n(lib_index_from_model_index(idx));
	layer.reset_to_default_color();
	lib_view().process_layer_stack();
}

void PVGuiQt::PVLayerStackModel::show_this_layer_only(const int idx)
{
	Inendi::PVLayerStack& layerstack = lib_layer_stack();
	int layer_idx = lib_index_from_model_index(idx);
	Inendi::PVLayer& layer = layerstack.get_layer_n(layer_idx);
	layer.set_visible(true); // in case, it isn't visible
	for (int i = 0; i < layerstack.get_layer_count(); i++) {
		if (i != layer_idx) {
			Inendi::PVLayer& layer = layerstack.get_layer_n(i);
			layer.set_visible(false);
		}
	}
	lib_view().process_layer_stack();
}

void PVGuiQt::PVLayerStackModel::layer_stack_refreshed()
{
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::add_new_layer(QString name)
{
	_lib_view.add_new_layer(name);
	Inendi::PVLayer& layer = lib_layer_stack().get_layer_n(rowCount() - 1);
	layer.reset_to_full_and_default_color();
	lib_view().process_layer_stack();
}

void PVGuiQt::PVLayerStackModel::move_selected_layer_up()
{
	beginResetModel();
	lib_view().move_selected_layer_up();
	lib_view().process_layer_stack();
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::move_selected_layer_down()
{
	beginResetModel();
	lib_view().move_selected_layer_down();
	lib_view().process_layer_stack();
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::delete_selected_layer()
{
	if (lib_layer_stack().get_selected_layer().is_locked()) {
		return;
	}

	_lib_view.delete_selected_layer();

	lib_view().process_layer_stack();
}

void PVGuiQt::PVLayerStackModel::duplicate_selected_layer(const QString& name)
{
	beginResetModel();
	lib_view().duplicate_selected_layer(name);
	lib_view().process_layer_stack();
	endResetModel();
}

void PVGuiQt::PVLayerStackModel::delete_layer_n(const int idx)
{
	assert(idx < rowCount());

	if (lib_layer_stack().get_layer_n(idx).is_locked()) {
		return;
	}

	_lib_view.delete_layer_n(idx);
	lib_view().process_layer_stack();
}
