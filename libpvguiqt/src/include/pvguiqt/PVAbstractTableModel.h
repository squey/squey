/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#ifndef PVGUIQT_PVABSTRACTTABLEMODEL_H
#define PVGUIQT_PVABSTRACTTABLEMODEL_H

#include <QAbstractTableModel>
#include <QBrush>

#include <pvbase/types.h>
#include <inendi/PVSelection.h>
#include <pvcop/db/array.h>

namespace PVGuiQt {

/**
 * This class replace QAbstractTableModel for big tables.
 * 
 * It handles huge number of rows using pagination.
 *
 * This model also handle sorting and filtering.
 *
 * @warning In Subclass functions, do not forget the rowIndex convertion to have correct row id
 *
 * user have to care about background display. If he want to handle selection,
 * he has to return the _selection_brush color on background for selected elements.
 */
class PVAbstractTableModel: public QAbstractTableModel {
	Q_OBJECT;

	public:
	/**
	 * Create a TableModel with a given number of row (default value)
	 */
	PVAbstractTableModel(int row_count, QObject* parent=nullptr);

	/**
	 * Function to export asked line;
	 */
	virtual QString export_line(int) const = 0;

	/// Selection

	/**
	 * Remove current selection
	 */
	void reset_selection();

	/**
	 * Start a selection at a given row.
	 *
	 * @param[in] row : Where we start the selection
	 */
	void start_selection(int row);

	/**
	 * Finish a selection at a given row.
	 *
	 * @param[in] row : Where the selection is over
	 */
	void end_selection(int row);

	/**
	 * Commit the "in progress" selection in the current selection.
	 *
	 * We use this mechanism to handle mouse movement during selection.
	 */
	void commit_selection();

	/**
	 * Wether a row is selected.
	 * 
	 * @note It care about selection in progress and current selection state.
	 */
	bool is_selected(QModelIndex const& index) const;

	/**
	 * Compute row number from a QModelIndex
	 *
	 *@param[in] index : index asked from view.
	 */
	int rowIndex(QModelIndex const& index) const;
	int rowIndex(PVRow index) const;

	/**
	 * Compute row position from Qt row position
	 *
	 * @note : use pagination information to get real position.
	 */
	int row_pos(QModelIndex const& index) const;
	int row_pos(PVRow index) const;

	/**
	 * Compute index from a row position.
	 *
	 * @note : apply filtering and sorting to row with pagination information.
	 */
	int row_pos_to_index(PVRow index) const;

	/**
	 * Number of ticks in the scrollbar
	 *
	 * @param index : Parent index (unused here)
	 * @return the number of scrollbar tick.
	 */
	int rowCount(const QModelIndex &index = QModelIndex()) const override final;

	/**
	 * Get real number of elements in the Table.
	 */
	size_t size() const { return _filter.size(); }

	/**
	 * Accessor for all lines in the ListingView
	 */
	std::vector<PVRow> const& shown_lines() const { return _filter; }

	/**
	 * Current_selection with possible modification
	 *
	 * @note: Modification is possible to enable Selection swapping
	 */
	Inendi::PVSelection & current_selection() { return _current_selection; }

	/// Accessors
	size_t current_page() const { return _current_page; }
	size_t& pos_in_page() { return _pos_in_page; }
	bool have_selection() const { return _start_sel != -1; }

	/**
	 * Move pagination information for many elements.
	 *
	 * @param[in] inc_elts : Number of elements to scroll
	 * @param[in] page_step : Number of elements in a view to handle last page
	 */
	void move_by(int inc_elts, size_t page_step);

	/**
	 * Move pagination to a given nraw id.
	 *
	 * This nraw id should be in selected elements
	 *
	 * @param[in] row : Row to scroll to
	 * @param[in] page_step : Number of elements in a view to handle last page
	 */
	void move_to_nraw(PVRow row, size_t page_step);

	/**
	 * Move pagination to a given row listing id.
	 *
	 * This row should be in the current listing selection.
	 *
	 * @param[in] row : Row to scroll to
	 * @param[in] page_step : Number of elements in a view to handle last page
	 */
	void move_to_row(PVRow row, size_t page_step);

	/**
	 * Move pagination to a given page
	 *
	 * Pagination is moved to the start of the page.
	 *
	 * @param[in] row : Row to scroll to
	 * @param[in] page_step : Number of elements in a view to handle last page
	 */
	void move_to_page(size_t page);

	/**
	 * Move pagination to the end of the listing
	 *
	 * @param[in] page_step : Number of elements in a view to handle last page
	 */
	void move_to_end(size_t page_step);

	/**
	 * Update pagination information with a given number of page.
	 *
	 * @note This should be called every time the number of pages change or
	 * when the number of elements in the listing change.
	 *
	 * @param[in] num_pages : Number of pages (scroll tick) in the listing
	 * @param[in] page_step : Number of elements in a view to handle last page
	 */
	void update_pages(size_t num_pages, size_t page_step);

	/**
	 * Check if we reach the end of the listing to get the last scrollbar tick.
	 */
	bool is_last_pos() const;

	pvcop::db::indexes& sorting() { return _sort; }

	/**
	 * Use default filtering with all elements from 0 to size;
	 */
	void reset_filter(int size);

        /**
	 * Remove filter. No more line displaied
	 */
	void clear_filter() { _filter.clear(); }

	/**
	 * Set the filter with the matching selection.
	 *
	 * @note perform convertion from selection to filter removing sort filtering.
	 */
	void set_filter(Inendi::PVSelection const* sel, size_t size);

	/**
	 * get index from filtered value.
	 *
	 * @note perform only sorting filtering.
	 */
	int filter_to_sort(PVRow row) const;

	protected:
	/**
	 * Set sorting contextual informations.
	 */
		void sorted(int col, Qt::SortOrder order);

	protected:
		/**
		 * Accessor for sorted column index.
		 */
		PVCol sorted_column() const { return _sorted_column; }

	protected:
	const QBrush _selection_brush = QColor(88, 172, 250);//!< Aspect of selected lines

	private:

	// Sorting information
	std::vector<PVRow> _filter; //!< Lines to use, map listing_row_id to nraw_row_id unsorted
	pvcop::db::indexes _sort; //!< Sorted lines, map listing not filtered position to nraw position
	PVCol _sorted_column; //!< The current sorted column
	Qt::SortOrder _sort_order; //!< The sort order of the current sorted column

	// Pagination information
	size_t _current_page; //!< Page currently processed
	size_t _pos_in_page; //!< Position in the page
	size_t _page_size; //!< Number of elements per page
	size_t _last_page_size; //!< Number of elements in the last page
	size_t _page_number; //!< Number of pages
	size_t _page_step; //!< Number of elements not counted in scroll ticks

	// Selection information
	Inendi::PVSelection _current_selection; //!< The current "visual" selection
	ssize_t _start_sel; //!< Begin of the "in progress" selection
	ssize_t _end_sel; //!< End of the "in progress" selection
	bool _in_select_mode; //!< Whether elements should be selected of unselected from "in progress" selection to current selection.
};

}
#endif
