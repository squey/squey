/**
 * @file
 * 
 * @copyright (C) ESI Group INENDI 2015
 */

#include <pvguiqt/PVAbstractTableModel.h>

namespace PVGuiQt {

// Adjustable number of ticks in the scrollbar
constexpr static size_t SCROLL_SIZE = 5000;
// Minimum number of elements per page to enable pagination
// It should be more than the maximum number of row we can display on a screen.
constexpr static size_t MIN_PAGE_SIZE = 100;

PVAbstractTableModel::PVAbstractTableModel(int row_count, QObject* parent):
	QAbstractTableModel(parent),
	_sort(row_count),
	_sorted_column(PVCOL_INVALID_VALUE),
	_sort_order(Qt::SortOrder::AscendingOrder),
	_current_page(0),
	_pos_in_page(0),
	_page_size(0),
	_last_page_size(0),
	_page_number(SCROLL_SIZE),
	_page_step(0),
	_start_sel(-1),
	_end_sel(-1),
	_in_select_mode(true)
{
	// No filter at start
	reset_filter(row_count);

	// No reorder at start
	auto& sort = _sort.to_core_array();
	std::iota(sort.begin(), sort.end(), 0);

	// Start with empty selection
	_current_selection.select_none();
}

/******************************************************************************
 *
 * PVAbstractTableModel::reset_filter
 *
 *****************************************************************************/
void PVAbstractTableModel::reset_filter(int size)
{
	// No filter at start
	_filter.resize(size);
	std::iota(_filter.begin(), _filter.end(), 0);
}

/******************************************************************************
 *
 * PVAbstractTableModel::reset_selection
 *
 *****************************************************************************/
void PVAbstractTableModel::reset_selection()
{
    _current_selection.select_none();
    _start_sel = _end_sel = -1;
}

/******************************************************************************
 *
 * PVAbstractTableModel::start_selection
 *
 *****************************************************************************/
void PVAbstractTableModel::start_selection(int row)
{
    assert(row != -1 && "Should be called only on checked row");
    _end_sel = _start_sel = row_pos(row);
    _in_select_mode = not _current_selection.get_line_fast(row_pos_to_index(_end_sel));
}

/******************************************************************************
 *
 * PVAbstractTableModel::end_selection
 *
 *****************************************************************************/
void PVAbstractTableModel::end_selection(int row)
{
    if(row != -1) {
	_end_sel = row_pos(row);
    }
}

/******************************************************************************
 *
 * PVAbstractTableModel::commit_selection
 *
 *****************************************************************************/
void PVAbstractTableModel::commit_selection()
{
    if(_end_sel == -1) {
	// No selection in progress
	return;
    }

    // Order begin and end of selection
    if(_end_sel < _start_sel) {
	std::swap(_start_sel, _end_sel);
    }

    // Update current_selection from "in progress" selection
    auto const& sort = _sort.to_core_array();
    for(; _start_sel<=_end_sel; _end_sel--) {
	_current_selection.set_line(row_pos_to_index(_end_sel), _in_select_mode);
    }

    // reset in progress selection
    _end_sel = _start_sel;

}


/******************************************************************************
 *
 * PVAbstractTableModel::rowIndex
 *
 *****************************************************************************/

int PVAbstractTableModel::rowIndex(QModelIndex const& index) const
{
	return rowIndex(index.row());
}

int PVAbstractTableModel::rowIndex(PVRow index) const
{
    // Compute index with : pagination information + offset from the start of
    // the "screen"
    // _filter convert listing line number to sorted nraw line number
    // _sort convert sorted nraw line number to nraw line number

	size_t idx = row_pos(index);

return row_pos_to_index(idx);
}

/******************************************************************************
 *
 * PVAbstractTableModel::row_pos_to_index
 *
 *****************************************************************************/
int PVAbstractTableModel::row_pos_to_index(PVRow idx) const
{
    return filter_to_sort(_filter[idx]);

}

/******************************************************************************
 *
 * PVAbstractTableModel::row_pos
 *
 *****************************************************************************/
int PVAbstractTableModel::row_pos(QModelIndex const& index) const
{
	return row_pos(index.row());
}

int PVAbstractTableModel::row_pos(PVRow index) const
{
    // Compute index with : pagination information + offset from the start of
    // the "screen"

	return (_current_page * _page_size + _pos_in_page) + (index - _current_page);
}


/******************************************************************************
 *
 * PVAbstractTableModel::rowCount
 *
 *****************************************************************************/
int PVAbstractTableModel::rowCount(const QModelIndex &) const
{
    // Define the number of ticks in the scrollbar
    if(_filter.size() > MIN_PAGE_SIZE * SCROLL_SIZE) {
	return _page_number + _page_step;
    } else {
	return _filter.size();
    }
}

/******************************************************************************
 *
 * PVAbstractTableModel::move_by
 *
 *****************************************************************************/

void PVAbstractTableModel::move_by(int inc_elts, size_t page_step)
{
    // Compute new position
    int new_pos = static_cast<int>(_pos_in_page) + inc_elts;

    // Reach next page but not the last one
    if(inc_elts > 0 and static_cast<size_t>(new_pos) >= _page_size and _current_page != _page_number - 1) {
	int incp = new_pos / _page_size; // Number of new page scrolled
	if(incp + _current_page >= _page_number)
	{
	    // Reach the end of the listing
	    _current_page = _page_number - 1;
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
    } else if((new_pos + _current_page * _page_size) >= (_filter.size() - page_step)) {
	// It is not the end of the last page but almost the end so we stop
	// now to show the last line at the bottom of the screen
	_current_page = _page_number - 1;
	_pos_in_page = std::max<int>(0, _last_page_size - page_step - 1);
    } else {
	// Scroll in the current page
	_pos_in_page = new_pos;
    }
}

/******************************************************************************
 *
 * PVAbstractTableModel::move_to_nraw
 *
 *****************************************************************************/
void PVAbstractTableModel::move_to_nraw(PVRow row, size_t page_step)
{
    // Row is line number in the full NRaw while line is the line number in
    // the current selection
    PVRow line = std::distance(_filter.begin(), std::find(_filter.begin(), _filter.end(), row));
    move_to_row(line, page_step);
}

/******************************************************************************
 *
 * PVAbstractTableModel::move_to_row
 *
 *****************************************************************************/
void PVAbstractTableModel::move_to_row(PVRow row, size_t page_step)
{
    assert(row< _filter.size() && "Impossible Row id");
    _current_page = row / _page_size;
    _pos_in_page = row - _current_page * _page_size;

    if(_current_page == _page_number) {
	// Do not scroll to much
	_pos_in_page = std::min(_pos_in_page, _last_page_size - page_step - 1);
    }
}

/******************************************************************************
 *
 * PVAbstractTableModel::move_to_page
 *
 *****************************************************************************/
void PVAbstractTableModel::move_to_page(size_t page)
{
    assert((page == 0 or page < _filter.size()) && "Impossible Row id");
    _current_page = page;
    _pos_in_page = 0;
}

/******************************************************************************
 *
 * PVAbstractTableModel::move_to_end
 *
 *****************************************************************************/
void PVAbstractTableModel::move_to_end(size_t page_step)
{
    _current_page = _page_number - 1;
    // It may happen that _last_page_size is 1 less than page_step du to
    // incomplete last row
    _pos_in_page = std::max<int>(0, _last_page_size - page_step - 1);
}

/******************************************************************************
 *
 * PVAbstractTableModel::update_pages
 *
 *****************************************************************************/
void PVAbstractTableModel::update_pages(size_t nbr_tick, size_t page_step)
{
    // Save pagination parameter to check for updates
    size_t old_page_num = _page_number;
    size_t old_step = _page_step;
    size_t old_last_page = _last_page_size;

    _page_step = page_step;
    // Filter may be updated before scrollbar
    assert(nbr_tick != 0 && "At least, there is the current page");
    if(_filter.size() > MIN_PAGE_SIZE * SCROLL_SIZE) {
	if(nbr_tick < SCROLL_SIZE / 2) {
	    // _filter is updated bu nbr_tick is not. Set a dummy value to
	    // initiate the fixed point algorithm and get correct page number
	    _page_size = _filter.size() / SCROLL_SIZE;
	} else {
	    // We keep the last tick for bottom
	    _page_size = _filter.size() / (nbr_tick - 1);
	}
	_page_number = _filter.size() / _page_size;
	// Last page is normal page + remainder
	_last_page_size = _filter.size() - _page_size * (_page_number - 1);
    } else {
	_page_size = 1;
	if(_page_step <= _filter.size()) {
	    _page_number = _filter.size() - _page_step + 1;
	} else {
	    _page_number = 1;
	}
	// Last page is normal page + remainder
	_last_page_size = _filter.size() - _page_number + 1;
    }

    if(old_page_num != _page_number or _page_step != old_step or
	    old_last_page != _last_page_size) {
	// Loop if we didn't reach a fixed point in pagination information
	emit layoutChanged();
    }
}

/******************************************************************************
 *
 * PVAbstractTableModel::is_last_pos
 *
 *****************************************************************************/
bool PVAbstractTableModel::is_last_pos() const
{
    return (_page_number - 1) == _current_page and
	_pos_in_page == static_cast<size_t>(std::max<int>(0, _last_page_size - _page_step - 1));
}

/******************************************************************************
 *
 * PVAbstractTableModel::is_selected
 *
 *****************************************************************************/
bool PVAbstractTableModel::is_selected(QModelIndex const& index) const
{
	int row_id = row_pos(index);
	int row = rowIndex(index);

	// Compute if line is in the "in progress" selection
	bool in_in_progress_sel = (_start_sel <= row_id and row_id <= _end_sel) or
		(_end_sel <= row_id and row_id <= _start_sel);
	// An element is selected if it is in curent_selection or
	// in "in progress" selection if we select new event
	// If we unselect event, an element is selected if it is
	// in current_selection but not in the "in progress" 
	// selection
	bool is_selected = (_in_select_mode and (in_in_progress_sel or _current_selection.get_line_fast(row)))
		or (not _in_select_mode and not in_in_progress_sel and _current_selection.get_line_fast(row));

	return is_selected;
}

/******************************************************************************
*
* PVAbstractTableModel::sorted
*
*****************************************************************************/
void PVAbstractTableModel::sorted(PVCol col, Qt::SortOrder order)
{
	_sorted_column = col;
	_sort_order = order;
}

/******************************************************************************
*
* PVAbstractTableModel::filter_to_sort
*
*****************************************************************************/
int PVAbstractTableModel::filter_to_sort(PVRow idx) const
{
	if (_sort_order == Qt::SortOrder::DescendingOrder) {
		idx = _sort.size() - idx -1;
	}

	const auto& sort = _sort.to_core_array();
    return sort[idx];

}

/******************************************************************************
*
* PVAbstractTableModel::set_filter
*
*****************************************************************************/
void PVAbstractTableModel::set_filter(Inendi::PVSelection const* sel, size_t size)
{
	auto const& sort = _sort.to_core_array();

	// Push selected lines
	for (PVRow line=0; line< size; line++) {
		// A line is selected if sorted one is in the selection.
		if (sel->get_line(sort[line])) {
			_filter.push_back(line);
		}
	}
}

}
