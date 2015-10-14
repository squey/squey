/**
 * \file PVListingModel.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QtCore>
#include <QtWidgets>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVColor.h>

#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>

#include <pvguiqt/PVCustomQtRoles.h>
#include <pvguiqt/PVListingModel.h>

/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVGuiQt::PVListingModel::PVListingModel(Picviz::PVView_sp& view, QObject* parent):
	QAbstractTableModel(parent),
	_zombie_brush(QColor(0, 0, 0)),
	_vheader_font(":/Convergence-Regular"),
	_view(view),
	_obs_vis(this),
	_obs_zomb(this)
{
	// Update the full model if axis combination change
	_obs_axes_comb.connect_refresh(this, SLOT(axes_comb_changed()));
	PVHive::get().register_observer(view, [=](Picviz::PVView& v) { return &v.get_axes_combination().get_axes_index_list(); },
		    			_obs_axes_comb);

	// Call update_filter on selection update
	_obs_sel.connect_refresh(this, SLOT(update_filter()));
	PVHive::get().register_observer(view, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, _obs_sel);

	// Update filter if we change layer content
	_obs_output_layer.connect_refresh(this, SLOT(update_filter()));
	PVHive::get().register_observer(view, [=](Picviz::PVView& view) { return &view.get_output_layer(); }, _obs_output_layer);

	// Update display of zombie lines on option toggling
	// FIXME : Can't we work without these specific struct?
	PVHive::get().register_func_observer(view, _obs_zomb);

	// Update display of unselected lines on option toogling
	// FIXME : Can't we work without these specific struct?
	PVHive::get().register_func_observer(view, _obs_vis);

	// No filter at start
	_filter.resize(lib_view().get_parent<Picviz::PVSource>()->get_row_count());
	std::iota(_filter.begin(), _filter.end(), 0);

	// No reorder at start
	_sort.resize(lib_view().get_parent<Picviz::PVSource>()->get_row_count());
	std::iota(_sort.begin(), _sort.end(), 0);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVGuiQt::PVListingModel::data(const QModelIndex &index, int role) const
{
	// Axis may have been duplicated and moved, get the real one.
	const PVCol org_col = lib_view().get_original_axis_index(index.column());

	switch (role) {
		// Get content of the cell
		case (Qt::DisplayRole):
			return _current_data;
		// Define alignment of data
		case (Qt::TextAlignmentRole):
			return {Qt::AlignLeft | Qt::AlignVCenter};
		// Define brackground color for cells
		case (Qt::BackgroundRole):
		{
			const PVRow r = rowIndex(index);
			if (lib_view().get_real_output_selection().get_line(r)) {
				// Selected elements, use output layer color
				const PVCore::PVHSVColor color = lib_view().get_color_in_output_layer(r);
				return QBrush(color.toQColor());
			} else if (lib_view().get_line_state_in_layer_stack_output_layer(r)) {
				/* The event is unselected use darker output layer color */
				const PVCore::PVHSVColor color = lib_view().get_color_in_output_layer(r);
				return QBrush(color.toQColor().darker(200));
			} else {
				/* The event is a ZOMBIE */
				return _zombie_brush;
			}
		}
		// Define Font color for cells
		case (Qt::ForegroundRole):
		{
			const PVRow r = rowIndex(index);
			// Show text in white if this is a zombie event
			if (!lib_view().get_real_output_selection().get_line(r) &&
				!lib_view().get_line_state_in_layer_stack_output_layer(r)) {
				return QBrush(Qt::white);
			}
			return QVariant();
		}
		// Define font to use
		case (Qt::FontRole):
		{
			// Set incomplete fields in italic
			QFont f;
			bool complete;
			// Ask data from NRaw
			_current_data = lib_view().get_parent<Picviz::PVSource>()->get_value(rowIndex(index), org_col, &complete);
			if(not complete) {
				f.setItalic(true);
			}

			return f;
		}
	}
	return QVariant();
}


/******************************************************************************
 *
 * PVGuiQt::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVGuiQt::PVListingModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	switch (role) {
		// Horizontal header contains axis labels and Vertical is line number
		case (Qt::DisplayRole):
			if (orientation == Qt::Horizontal) {
				if (section >= 0) {
					return lib_view().get_axis_name(section);
				}
			} else if (section >= 0) {
				assert(orientation == Qt::Vertical && "No others possible orientations.");
				return rowIndex(section);
			}
			break;
		// Selected lines are bold, others use class specific font
		case (Qt::FontRole):
			if (orientation == Qt::Vertical and section >= 0) {
				if (lib_view().get_real_output_selection().get_line(section)) {
					QFont f(_vheader_font);
					f.setBold(true);
					return f;
				}
				return _vheader_font;
			}
			break;
		// Define header alignment
		case (Qt::TextAlignmentRole):
			if (orientation == Qt::Horizontal) {
				return (Qt::AlignLeft + Qt::AlignTop);
			} else {
				return (Qt::AlignRight + Qt::AlignVCenter);
			}
	}

	return QVariant();
}


/******************************************************************************
 *
 * PVGuiQt::PVListingModel::rowCount
 *
 *****************************************************************************/
int PVGuiQt::PVListingModel::rowCount(const QModelIndex &) const
{
	return _filter.size();
}


/******************************************************************************
 *
 * PVGuiQt::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVGuiQt::PVListingModel::columnCount(const QModelIndex &) const
{
	return lib_view().get_column_count();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVGuiQt::PVListingModel::flags(const QModelIndex &/*index*/) const
{
	// TODO : Should not be selectable
	return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::axes_comb_changed
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::axes_comb_changed()
{
	// Inform others widgets model is reset and view have to be reloaded
	beginResetModel();
	endResetModel();
}


/******************************************************************************
 *
 * PVGuiQt::PVListingModel::rowIndex
 *
 *****************************************************************************/

int PVGuiQt::PVListingModel::rowIndex(QModelIndex const& index) const
{
	return rowIndex(index.row());
}

int PVGuiQt::PVListingModel::rowIndex(PVRow index) const
{
	return _sort[_filter[index]];
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::sort
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::sort(PVCol col, Qt::SortOrder order, tbb::task_group_context & ctxt)
{
	lib_view().sort_indexes_with_axes_combination(col, order, _sort, &ctxt);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::update_filter
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::update_filter()
{
	Picviz::PVSelection const* sel = lib_view().get_selection_visible_listing();
	// Everything is selected
	if(not sel) {
		_filter.resize(lib_view().get_row_count());
		std::iota(_filter.begin(), _filter.end(), 0);
		return;
	}

	// Filter out lines according to the good selection.
	_filter.clear();
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, lib_view().get_row_count());
	// Nothing is visible
	if (nvisible_lines == 0) {
		return;
	}

	// Inform view about future update
	emit layoutAboutToBeChanged();

	// Push selected lines
	for (PVRow line=0; line< lib_view().get_row_count(); line++) {
		if (sel->get_line(line)) {
			_filter.push_back(line);
		}
	}

	// Inform view new_filter is set
	// This is not done using Hive as _filter have to be set, PVSelection is not
	// enough
	emit layoutChanged();
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVListingVisibilityObserver::update
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVListingVisibilityObserver::update(arguments_type const&) const
{
    _parent->update_filter();
}

/******************************************************************************
 *
 * PVGuiQt::__impl::PVListingVisibilityZombieObserver::update
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVListingVisibilityZombieObserver::update(arguments_type const&) const
{
    _parent->update_filter();
}
