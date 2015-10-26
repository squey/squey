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

// Maximum number of ticks in the scroll bar
constexpr static size_t SCROLL_SIZE = 5000;

/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVGuiQt::PVListingModel::PVListingModel(Picviz::PVView_sp& view, QObject* parent):
	QAbstractTableModel(parent),
	_zombie_brush(QColor(0, 0, 0)),
	_selection_brush(QColor(88, 172, 250)),
	_vheader_font(":/Convergence-Regular"),
	_view(view),
	_obs_vis(this),
	_obs_zomb(this),
	_current_page(0),
	_pos_in_page(0),
	_page_size(0),
	_last_page_size(0),
	_start_sel(-1),
	_end_sel(-1),
	_in_select_mode(true)
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

	// Start with empty selection
	_current_selection.select_none();
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
		    {
			const PVRow r = rowIndex(index);
			if(r >= lib_view().get_row_count()) {
			    // This data should not be shown as it is not in
			    // the NRaw
			    return {};
			}
			return _current_data;
		    }
		// Define alignment of data
		case (Qt::TextAlignmentRole):
			return {Qt::AlignLeft | Qt::AlignVCenter};
		// Get Tooltip content for the cell
		case Qt::ToolTipRole:
			{
			    const PVRow r = rowIndex(index);
			    return lib_view().get_parent<Picviz::PVSource>()->get_value(r, org_col);
			}
		// Define brackground color for cells
		case (Qt::BackgroundRole):
		{
			const PVRow r = rowIndex(index);
			if(r >= lib_view().get_row_count()) {
			    // Nothing for rows out of bound.
			    return {};
			}

			// Compute if line is in the "in progress" selection
			bool in_in_progress_sel = (_start_sel <= r and r <= _end_sel) or
			    (_end_sel <= r and r <= _start_sel);
			// An element is selected if it is in curent_selection or
			// in "in progress" selection if we select new event
			// If we unselect event, an element is selected if it is
			// in current_selection but not in the "in progress" 
			// selection
			bool is_selected = (_in_select_mode and (in_in_progress_sel or _current_selection.get_line_fast(r)))
			    or (not _in_select_mode and not in_in_progress_sel and _current_selection.get_line_fast(r));

			if(is_selected) {
				// Visual selected lines from current selection
				// and "in progress" selection
				return _selection_brush;
			} else if (lib_view().get_real_output_selection().get_line(r)) {
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
			if(r >= lib_view().get_row_count()) {
			    // Nothing for rows out of bound.
			    return {};
			}
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
			const PVRow r = rowIndex(index);
			if(r >= lib_view().get_row_count()) {
			    return {};
			}
			// Ask data from NRaw
			bool complete;
			_current_data = lib_view().get_parent<Picviz::PVSource>()->get_value(r, org_col, &complete);
			QFont f;
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
    // Define the number of ticks in the scrollbar
    return std::min<int>(_filter.size(), SCROLL_SIZE);
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
	return Qt::ItemIsEnabled;
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
    // Compute index with : pagination information + offset from the start of
    // the "screen"
    // _filter convert listing line number to sorted nraw line number
    // _sort convert sorted nraw line number to nraw line number
    return _sort[_filter[(_current_page * _page_size + _pos_in_page) + (index - _current_page)]];
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
	// Reset the current selection as context changed
	reset_selection();

	Picviz::PVSelection const* sel = lib_view().get_selection_visible_listing();

	// Inform view about future update
	emit layoutAboutToBeChanged();

	// Everything is selected
	if(not sel) {
		_filter.resize(lib_view().get_row_count());
		std::iota(_filter.begin(), _filter.end(), 0);
		emit layoutChanged(); // FIXME : Should use RAII
		return;
	}

	// Filter out lines according to the good selection.
	_filter.clear();
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, lib_view().get_row_count());
	// Nothing is visible
	if (nvisible_lines == 0) {
	    emit layoutChanged(); // FIXME : Should use RAII
	    return;
	}

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
 * PVGuiQt::PVListingModel::max_page
 *
 *****************************************************************************/
size_t PVGuiQt::PVListingModel::max_page() const
{
    return (_filter.size() - _last_page_size) / _page_size;
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::start_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::start_selection(int row)
{
    assert(row != -1 && "Should be called only on checked row");
    _end_sel = _start_sel = rowIndex(row);
    _in_select_mode = not _current_selection.get_line_fast(_end_sel);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::end_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::end_selection(int row)
{
    if(row != -1) {
	_end_sel = rowIndex(row);
    }
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::commit_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::commit_selection()
{
    if(_end_sel == -1) {
	// No selection in progress
	return;
    }

    // Order begin and end of selection
    if(_end_sel < _start_sel) {
	std::swap(_start_sel, _end_sel);
    }

    auto start_filter = std::find(_filter.begin(), _filter.end(), _start_sel);
    if(_start_sel == -1)
    {
	start_filter = _filter.begin();
    }
    auto end_filter = std::find(start_filter, _filter.end(), _end_sel);

    // Update current_selection from "in progress" selection
    for(; start_filter<=end_filter; start_filter++) {
	_current_selection.set_line(*start_filter, _in_select_mode);
    }

    // reset in progress selection
    _end_sel = _start_sel;

}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::move_by
 *
 *****************************************************************************/

void PVGuiQt::PVListingModel::move_by(int inc_elts, size_t page_step)
{
    const size_t max_page = this->max_page();

    // Compute new position
    int new_pos = static_cast<int>(_pos_in_page) + inc_elts;

    // Reach next page but not the last one
    if(inc_elts > 0 and static_cast<size_t>(new_pos) > _page_size and _current_page != max_page) {
	int incp = new_pos / _page_size; // Number of new page scrolled
	if(incp + _current_page > max_page)
	{
	    // Reach the end of the listing
	    _current_page = max_page;
	    _pos_in_page = _last_page_size - page_step;
	} else {
	    // Go to the correct page
	    _current_page += incp;
	    _pos_in_page = new_pos - incp * _page_size;
	}
    } else if(inc_elts < 0 and new_pos < 0) {
	// Reach previous page
	// Number of page scroll back
	// -1 as we keep positif _pos_in_page
	int decp = new_pos / static_cast<int>(_page_size) - 1;
	if((decp + static_cast<int>(_current_page)) < 0) {
	    // Reach the start of the listing
	    _current_page = 0;
	    _pos_in_page = 0;
	} else {
	    // go to the correct previous page
	    _current_page += decp;
	    _pos_in_page = new_pos - decp * _page_size;
	}
    } else if((new_pos + _current_page * _page_size) > (_filter.size() - page_step)) {
	// It is not the end of the last page but almost the end so we stop
	// now to show the last line at the bottom of the screen
	_current_page = max_page;
	_pos_in_page = _last_page_size - page_step;
    } else {
	// Scroll in the current page
	_pos_in_page = new_pos;
    }
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::move_to_nraw
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::move_to_nraw(PVRow row, size_t page_step)
{
    // Row is line number in the full NRaw while line is the line number in
    // the current selection
    PVRow line = std::distance(_filter.begin(), std::find(_filter.begin(), _filter.end(), row));
    move_to_row(line, page_step);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::move_to_row
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::move_to_row(PVRow row, size_t page_step)
{
    assert(row< _filter.size() && "Impossible Row id");
    _current_page = row / _page_size;
    _pos_in_page = row - _current_page * _page_size;

    if(_current_page == max_page()) {
	// Do not scroll to much
	_pos_in_page = std::min(_pos_in_page, _last_page_size - page_step);
    }
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::move_to_page
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::move_to_page(size_t page)
{
    assert(page< _filter.size() && "Impossible Row id");
    _current_page = page;
    _pos_in_page = 0;
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::move_to_end
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::move_to_end(size_t page_step)
{
    _current_page = max_page();
    // It may happen that _last_page_size is 1 less than page_step du to
    // incomplete last row
    _pos_in_page = std::max<int>(0, _last_page_size - page_step);
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::update_pages
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::update_pages(size_t num_pages, size_t page_step)
{
    // With filter.size() < page_step, num_pages should be 0 but pages may be
    // update before QtableView update for the verticalScrollbar
    if(num_pages == 0 or _filter.size() < page_step) {
	_page_size = 1;
	_last_page_size = std::min(_filter.size(), page_step);
    } else {
	// filter.size() - page_step < num_pages is a strange bug du to 
	// incomplete last row ...
	if((_filter.size() - page_step) < num_pages) {
	    _page_size = 1;
	} else {
	    _page_size = (_filter.size() - page_step) / num_pages;
	}
	_last_page_size = _filter.size() - _page_size * num_pages;
    }
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::reset_selection
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::reset_selection()
{
    _current_selection.select_none();
    _start_sel = _end_sel = -1;
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
