/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#include <pvguiqt/PVAbstractTableModel.h>

#include <omp.h>

namespace PVGuiQt
{

// Adjustable number of ticks in the scrollbar
constexpr static size_t SCROLL_SIZE = 5000;
// Minimum number of elements per page to enable pagination
// It should be more than the maximum number of row we can display on a screen.
constexpr static size_t MIN_PAGE_SIZE = 100;

PVAbstractTableModel::PVAbstractTableModel(int row_count, QObject* parent)
    : QAbstractTableModel(parent)
    , _display(row_count)
    , _current_page(0)
    , _pos_in_page(0)
    , _page_size(0)
    , _last_page_size(0)
    , _page_number(SCROLL_SIZE)
    , _page_step(0)
    , _current_selection(row_count)
    , _start_sel(-1)
    , _end_sel(-1)
    , _in_select_mode(true)
    , _selection_mode(SET)
{
	// Start with empty selection
	_current_selection.select_none();
}

/******************************************************************************
 *
 * PVAbstractTableModel::set_selection_mode
 *
 *****************************************************************************/

void PVAbstractTableModel::set_selection_mode(selection_mode_t mode)
{
	_selection_mode = mode;
}

/******************************************************************************
 *
 * PVAbstractTableModel::reset_selection
 *
 *****************************************************************************/
void PVAbstractTableModel::reset_selection()
{
	_current_selection.select_none();
	clear_selection();
}

/******************************************************************************
 *
 * PVAbstractTableModel::clear_selection
 *
 *****************************************************************************/
void PVAbstractTableModel::clear_selection()
{
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
	_in_select_mode = not _current_selection.get_line_fast(_display.row_pos_to_index(_end_sel));
}

/******************************************************************************
 *
 * PVAbstractTableModel::end_selection
 *
 *****************************************************************************/
void PVAbstractTableModel::end_selection(int row)
{
	if (row != -1) {
		if (_start_sel == -1) {
			/* if the range selection has been previously reset, doing a shift+left
			 * mouse button
			 * in PVAbstractTableView will call this method; _start_sel must also be
			 * initialized
			 * to 0 to have a valid range selection in compliance with QTableView
			 * behaviour.
			 */
			_start_sel = 0;
		}
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
	if (_end_sel == -1) {
		// No selection in progress
		return;
	}

	// Order begin and end of selection
	if (_end_sel < _start_sel) {
		std::swap(_start_sel, _end_sel);
	}

	// Update current_selection from "in progress" selection
	for (; _start_sel <= _end_sel; _end_sel--) {
		int index = _display.row_pos_to_index(_end_sel);
		bool is_set = _current_selection.get_line_fast(index);
		_current_selection.set_line(index, apply_selection_mode(is_set));
	}

	// reset in progress selection
	_end_sel = _start_sel;
}

/******************************************************************************
 *
 * PVAbstractTableModel::has_selection
 *
 *****************************************************************************/

bool PVAbstractTableModel::has_selection() const
{
	return (not _current_selection.is_empty()) || (_end_sel != -1);
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

	size_t idx = row_pos(index);

	return _display.row_pos_to_index(idx);
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
int PVAbstractTableModel::rowCount(const QModelIndex&) const
{
	// Define the number of ticks in the scrollbar
	if (_display.size() > MIN_PAGE_SIZE * SCROLL_SIZE) {
		return _page_number + _page_step;
	} else {
		return _display.size();
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
	int new_pos = int(_pos_in_page) + inc_elts;

	// Reach next page but not the last one
	if (inc_elts > 0 and size_t(new_pos) >= _page_size and _current_page != _page_number - 1) {
		int incp = new_pos / _page_size; // Number of new page scrolled
		if (incp + _current_page >= _page_number) {
			// Reach the end of the listing
			_current_page = _page_number - 1;
			_pos_in_page = _last_page_size - page_step;
		} else {
			// Go to the correct page
			_current_page += incp;
			_pos_in_page = new_pos - incp * _page_size;
		}
	} else if (inc_elts < 0 and new_pos < 0) {
		// Reach previous page
		// Number of page scroll back
		int decp = new_pos / int(_page_size) - 1;
		if ((decp + int(_current_page)) < 0) {
			// Reach the start of the listing
			_current_page = 0;
			_pos_in_page = 0;
		} else {
			// go to the correct previous page
			_current_page += decp;
			_pos_in_page = std::min(new_pos - decp * _page_size, _page_size - 1);
		}
	} else if ((new_pos + _current_page * _page_size) >= (_display.size() - page_step)) {
		// It is not the end of the last page but almost the end so we stop
		// now to show the last line at the bottom of the screen
		_current_page = _page_number - 1;
		_pos_in_page = 0;
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
	PVRow line = _display.row_pos_from_index(row);
	move_to_row(line, page_step);
}

/******************************************************************************
 *
 * PVAbstractTableModel::move_to_row
 *
 *****************************************************************************/
void PVAbstractTableModel::move_to_row(PVRow row, size_t page_step)
{
	assert(row < _display.size() && "Impossible Row id");
	_current_page = row / _page_size;
	_pos_in_page = row - _current_page * _page_size;

	if (_current_page == _page_number) {
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
	assert((page == 0 or page < _display.size()) && "Impossible Row id");
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
	_pos_in_page = std::max<int>(0, int(_last_page_size) - int(page_step) - 1);
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

	if (nbr_tick == 1 and page_step != _display.size()) {
		// _display may be updated while nbr_tck and page_step are not. Set a dummy
		// information for the first loop then it will be updated again on the fixed
		// point computation.
		_page_step = std::min(SCROLL_SIZE, _display.size());
	} else {
		_page_step = page_step;
	}
	// Filter may be updated before scrollbar
	assert(nbr_tick != 0 && "At least, there is the current page");
	if (_display.size() > MIN_PAGE_SIZE * SCROLL_SIZE) {
		if (nbr_tick < SCROLL_SIZE / 2 or nbr_tick > _display.size()) {
			// _display is updated but nbr_tick is not. Set a dummy value to
			// initiate the fixed point algorithm and get correct page number
			_page_size = _display.size() / SCROLL_SIZE;
		} else {
			// We keep the last tick for bottom
			_page_size = _display.size() / (nbr_tick - 1);
		}
		_page_number = _display.size() / _page_size;
		// Last page is normal page + remainder
		_last_page_size = _display.size() - _page_size * (_page_number - 1);
	} else {
		_page_size = 1;
		if (_page_step <= _display.size()) {
			_page_number = _display.size() - _page_step + 1;
		} else {
			_page_number = 1;
		}
		// Last page is normal page + remainder
		_last_page_size = _display.size() - _page_number + 1;
	}

	if (old_page_num != _page_number or _page_step != old_step or
	    old_last_page != _last_page_size) {
		// Loop if we didn't reach a fixed point in pagination information
		Q_EMIT layoutChanged();
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
	       _pos_in_page ==
	           static_cast<size_t>(std::max<int>(0, int(_last_page_size) - int(_page_step) - 1));
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
	bool is_selected = _current_selection.get_line_fast(row);
	bool is_in_progress_sel = (_start_sel <= row_id and row_id <= _end_sel) or
	                          (_end_sel <= row_id and row_id <= _start_sel);

	if (is_in_progress_sel) {
		is_selected = apply_selection_mode(is_selected);
	}

	return is_selected;
}

/******************************************************************************
*
* PVAbstractTableModel::sorted
*
*****************************************************************************/
void PVAbstractTableModel::sorted(PVCombCol col, Qt::SortOrder order)
{
	_display.set_sorted_meta(col, order);
	// Commit the range selection to make the selected rows persistent
	commit_selection();
	// And reset the range selection which does not have sense anymore.
	_end_sel = _start_sel = -1;
}

/******************************************************************************
*
* PVAbstractTableModel::get_wrapped_string
*
*****************************************************************************/
QString PVAbstractTableModel::get_wrapped_string(const QString& str) const
{
	static constexpr const int WORDWRAP_SIZE = 200;

	QString res = str;

	for (int i = WORDWRAP_SIZE; i < res.size(); i += WORDWRAP_SIZE) {
		res.insert(i, "<br>");
	}
	return res;
}

/******************************************************************************
*
* PVAbstractTableModel::apply_selection_mode
*
*****************************************************************************/

bool PVAbstractTableModel::apply_selection_mode(bool value) const
{
	if (_selection_mode == PVAbstractTableModel::SET) {
		return true;
	} else if (_selection_mode == PVAbstractTableModel::TOGGLE_AND_USE) {
		return _in_select_mode;
	} else { // PVAbstractTableModel::NEGATE
		return not value;
	}
}
} // namespace PVGuiQt
