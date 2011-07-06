//! \file PVListingView.h
//! $Id: PVListingView.h 3092 2011-06-09 06:43:23Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLISTINGVIEW_H
#define PVLISTINGVIEW_H

#include <QTableView>
#include <QHeaderView>

#include <picviz/PVView.h>

#include <PVListingModel.h>
#include <PVLayerStackModel.h>
#include <PVListingNoUnselectedModel.h>
#include <PVListingNoZombieModel.h>
#include <PVListingNoZombieNoUnselectedModel.h>


namespace PVInspector {
class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVListingView
 */
class PVListingView : public QTableView
{
	Q_OBJECT

	PVMainWindow    *main_window; //<!
	Picviz::PVView_p lib_view;    //<!

/*	PVListingModel                     *pv_listing_model;                         //!<
	PVListingNoUnselectedModel         *pv_listing_no_unselected_model;           //!<
	PVListingNoZombieModel             *pv_listing_no_zombie_model;               //!<
	PVListingNoZombieNoUnselectedModel *pv_listing_no_zombie_no_unselected_model; //!<
	int screenshot_index;*/

	public slots:
		void slotDoubleClickOnVHead(int);
// FIXME!			void update_row_count_in_all_dynamic_listing_model_Slot();
			/* void update_to_current_selection_Slot();*/

/*			void refresh_listing_Slot();
			void refresh_listing_with_horizontal_header_Slot();
      void selection_changed_Slot();
			void update_pv_listing_model_Slot();*/

public:
// Removed (=> PVTabSplitter) PVLayerStackModel *pv_layer_stack_model; // FIXME: should be private

	/**
	 * Constructor.
	 *
	 * @param mw
	 * @param lib_view
	 * @param parent
	 */
	PVListingView(PVMainWindow *mw, Picviz::PVView_p lib_view, PVTabSplitter *parent);

	/**
	 *
	 * @return
	 */
//	picviz_view_t *get_lib_view()const{return lib_view;}

	/**
	 *
	 * @return
	 */
//	int get_screenshot_index();

	/**
	 *
	 */
//	void increment_screenshot_index();

private:
	void mouseReleaseEvent(QMouseEvent *event);

};
}

#endif // PVLISTINGVIEW_H
