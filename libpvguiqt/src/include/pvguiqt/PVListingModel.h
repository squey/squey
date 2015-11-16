/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVLISTINGMODEL_H
#define PVGUIQT_PVLISTINGMODEL_H

#include <vector>
#include <utility>

#include <QAbstractTableModel>
#include <QBrush>
#include <QFont>

#include <pvkernel/core/general.h>
#include <inendi/PVAxesCombination.h>
#include <inendi/PVView_types.h>

#include <pvhive/PVObserverSignal.h>

#include <pvhive/PVActor.h>

namespace tbb {
    class task_group_context;
}

namespace PVGuiQt {

    class PVListingModel;

namespace __impl {
    /**
     * PVListingVisibilityObserver
     *
     * This dummy class is used to look at the toggle_listing_unselected_visibility
     * function call to update filter in the ListingModel
     *
     * @note Hive inside
     *
     */
    struct PVListingVisibilityObserver: public PVHive::PVFuncObserver<Inendi::PVView, FUNC(Inendi::PVView::toggle_listing_unselected_visibility)>
    {
	/**
	 * Save the ListingModel to be updated.
	 *
	 * @param parent : ListingModel to update.
	 */
	PVListingVisibilityObserver(PVGuiQt::PVListingModel* parent):
	    _parent(parent)
	{ }

	private:
	/**
	 * Update the ListingModel update.
	 *
	 * @param args : None
	 */
	void update(arguments_type const&) const override;

	private:
	PVGuiQt::PVListingModel* _parent; //!< ListingModel to update.
    };

    /**
     * PVListingVisibilityObserver
     *
     * This dummy class is used to look at the toggle_listing_zombie_visibility
     * function call to update filter in the ListingModel
     *
     * @note Hive inside
     *
     */
    struct PVListingVisibilityZombieObserver: public PVHive::PVFuncObserver<Inendi::PVView, FUNC(Inendi::PVView::toggle_listing_zombie_visibility)>
    {
	/**
	 * Save the ListingModel to be updated.
	 *
	 * @param parent : ListingModel to update.
	 */
	PVListingVisibilityZombieObserver(PVGuiQt::PVListingModel* parent):
	    _parent(parent)
	{ }

	private:
	/**
	 * Update the ListingModel update.
	 *
	 * @param args : None
	 */
	void update(arguments_type const& args) const override;

	private:
	PVGuiQt::PVListingModel* _parent; //!< ListingModel to update.
    };

}

/**
 * \class PVListingModel
 *
 * Model to display an NRaw table.
 *
 * It supports Selected, Unselected and Zombie lines.
 */

class PVListingModel : public QAbstractTableModel
{
    Q_OBJECT

    /// Graphical information

    // These data are graphical but have to be in the model because in Qt MVC
    // pattern, the model is responsive to give this information using roles
    // status.

private:
	QBrush _zombie_brush;	//!< Aspect of zombie lines
	QBrush _selection_brush;//!< Aspect of selected lines
	QFont  _vheader_font;	//!< Font for header view

public:
    /**
     * Create a Listing model.
     *
     * Initialise communication with others widgets and set graphical aspect.
     *
     * @param view : Global display for data informations
     * @param parent : Parent widget
     *
     * @note It use a view as a parameter to register observer. Thanks to this
     * record, every view will be updated on listing model modification.
     *
     */
    PVListingModel(Inendi::PVView_sp& view, QObject* parent = nullptr);

    /**
     * Return data requested by the View
     *
     * Provided data are value content,text alignment, background and forground
     * colors and font
     *
     * @param index : Cell asked for information.
     * @param role : Kind of information required.
     * @return Matching information depending on role (QVariant)
     *
     * @warning: Do not use DisplayRole as a default because this function can be
     * call only through the view. Ask to the nraw if you want information.
     */
    QVariant data(const QModelIndex &index, int role) const override;

    /**
     * return header information for given section.
     *
     * Returned informations are value, font and alignment.
     *
     * @param section : line number or colonne number
     * @param orientation : Which header to concider. It is use with section to
     * determine correct header.
     * @param role : What kind of information is required.
     * @return Header information depending on role. (QVariant)
     */
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

    /**
     * Number of ticks in the scrollbar
     *
     * @param index : Parent index (unused here)
     * @return the number of scrollbar tick.
     */
    int rowCount(const QModelIndex &index = QModelIndex()) const override;

    /**
     * Compute row number from a QModelIndex
     *
     *@param[in] index : index asked from view.
     */
    int rowIndex(QModelIndex const& index) const;
    int rowIndex(PVRow index) const;

    /**
     * Number of column in the view
     *
     * @param index : Parent index (unused here)
     * @return The number of axis in the view.
     */
    int columnCount(const QModelIndex &index = QModelIndex()) const override;

    /**
     * Define possible interactions with the model.
     *
     * @param index : flags can be partial. (Unused here)
     *
     * @return model flags.
     */
    Qt::ItemFlags flags(const QModelIndex &index) const override;

    /**
     * Sort the Listing on a given column and order.
     *
     * @param[in] col : Column to sort.
     * @param[in] order : Order to use for sorting.
     * @param[in,out] ctxt : Information about sorting processing to enable cancel.
     *
     */
    void sort(PVCol col, Qt::SortOrder order, tbb::task_group_context & ctxt);

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

    private slots:
	/**
	 * With axes combination modifications, we have to update the model and
	 * reorder/add/remove columns.
	 */
    void axes_comb_changed();

    public slots:
	// public slots call through Hive
	/**
	 *  Update the current filter to show selected lines only.
	 */
    void update_filter();

    private:
    /**
     * Get the view linked with the listingView
     *
     * @return linked view.
     */
    inline Inendi::PVView const& lib_view() const { return *_view; }

    private:
    Inendi::PVView_sp _view; //!< Observed view
    PVHive::PVObserverSignal<Inendi::PVAxesCombination::columns_indexes_t> _obs_axes_comb; //!< Observe axs combination modifications
    PVHive::PVObserverSignal<Inendi::PVSelection> _obs_sel; //!< Observe the seletion to update on selection modifications
    PVHive::PVObserverSignal<Inendi::PVLayer> _obs_output_layer; //!< Observe selected/unselected calques
    __impl::PVListingVisibilityObserver _obs_vis; //!< Observer for selected/unselected lines
    __impl::PVListingVisibilityZombieObserver _obs_zomb; //! Observer for zombies lines

    // We save the current data to avoid asking for it twice in the NRaw.
    // We ask for NRaw value at FontRole time and re-use it at Display time
    // which is called right after the fontrole.
    // We use mutable as the data Qt interface function have to be const
    mutable QString _current_data; //!< Data of the current cell.
    std::vector<PVRow> _filter; //!< Lines to use, map listing_row_id to nraw_row_id unsorted
    std::vector<PVRow> _sort; //!< Sorted lines, map listing not filtered position to nraw position

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
