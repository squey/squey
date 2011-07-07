//! \file PVTabSplitter.h
//! $Id: PVTabSplitter.h 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVTABSPLITTER_H
#define PVTABSPLITTER_H


#include <QSplitter>

#include <picviz/PVView.h>

#include <PVListingView.h>
#include <PVListingModel.h>
#include <PVListingNoUnselectedModel.h>
#include <PVListingNoZombieModel.h>
#include <PVListingNoZombieNoUnselectedModel.h>

#include <PVLayerStackModel.h>
#include <PVLayerStackWidget.h>

#include <PVExtractorWidget.h>

namespace PVInspector {
class PVMainWindow;

/**
 *  \class PVTabSplitter
 */
class PVTabSplitter : public QSplitter
{
	Q_OBJECT

	PVMainWindow     *main_window;   //!< The parent PVMainWindow of this PVTabSplitter
	Picviz::PVView_p lib_view;      //!< The Picviz::PVView 

	PVListingView *pv_listing_view; //!< The PVListingView attached with our main application

        PVListingModel *pv_listing_model; //!< The classical Listing model (with zombies and unselected)
        PVListingModel *pv_listing_no_unselected_model; //!< The Listing model with zombies and without unselected
        PVListingModel *pv_listing_no_zombie_model; //!< The Listing model without zombies but with unselected
        PVListingModel *pv_listing_no_zombie_no_unselected_model; //!< The Listing model without both zombies and unselected

	PVLayerStackModel  *pv_layer_stack_model;
	PVLayerStackWidget *pv_layer_stack_widget;

	PVExtractorWidget *_pv_extractor; //!< The extractor widget of this view

	int screenshot_index;
	QString _tab_name;

public slots:
// FIXME!			void update_row_count_in_all_dynamic_listing_model_Slot();
			/* void update_to_current_selection_Slot();*/

	/**
	 * The Slot that will refresh the content of the PVListingView
	 */
	void refresh_listing_Slot();
	
	/**
	 * The Slot that will refresh the PVListingView with it's horizontal header
	 */
	void refresh_listing_with_horizontal_header_Slot();
	
	/**
	 *
	 */
	void selection_changed_Slot();
	
	/**
	 *
	 */
	void update_pv_listing_model_Slot();

public:
	/**
	 * Constructor.
	 *
	 * @param mw
	 * @param lib_view
	 * @param parent
	 */
	PVTabSplitter(PVMainWindow *mw, Picviz::PVView_p lib_view, QString const& tab_name, QWidget *parent);

	virtual ~PVTabSplitter();

	/**
	 *
	 * @return a pointer to the PVLayerStackModel attached to this PVMainSplitter
	 */
	PVLayerStackModel  *get_layer_stack_model()const{return pv_layer_stack_model;}

	/**
	 *
	 * @return a pointer to the PVLayerStackWidget attached to this PVMainSplitter
	 */
	PVLayerStackWidget *get_layer_stack_widget()const{return pv_layer_stack_widget;}
	
	/**
	 *
	 * @return a pointer to the Picviz::PVView attached to this PVMainSplitter
	 */
	Picviz::PVView_p get_lib_view()const{return lib_view;}
	
	/**
	 *
	 * @return a pointer to the Picviz::PVMainWindow attached to this PVMainSplitter
	 */
	PVMainWindow* get_main_window() const { return main_window; }
 	
	/**
	 *
	 * @return a pointer to the PVInspector::PVExtractorWidget attached to this PVMainSplitter
	 */
	PVExtractorWidget* get_extractor_widget() const {return _pv_extractor;}

	QString get_tab_name() const { return _tab_name; }

	/**
	 *
	 * @return the index of the next screenshot
	 */
	int get_screenshot_index();

	/**
	 * Increments the index of the next screenshot
	 */
	void increment_screenshot_index();
	
	/**
	* Update filter menu enabling
	*/
	void updateFilterMenuEnabling();
	
public slots:
	/**
	 * The Slot that will refresh the PVLayerStackWidget
	 */
	void refresh_layer_stack_view_Slot(); // From PVLayerStackWindow
//		void refresh();
//		void update_pv_layer_stack_model_Slot();
signals:
	/**
	* The selection has changed
	*/
	void selection_changed_signal(bool);
};
}

#endif
