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

#include <QBrush>
#include <QFont>

#include <pvguiqt/PVAbstractTableModel.h>
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

class PVListingModel : public PVAbstractTableModel
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
     * Export row-th line in a QString.
     * 
     * @param row: Element to export.
     * @return row-th line as a QString.
     */
    QString export_line(int row) const override;

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

};

}

#endif
