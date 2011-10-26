//! \file PVTabSplitter.h
//! $Id: PVTabSplitter.h 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVTABSPLITTER_H
#define PVTABSPLITTER_H


#include <QSplitter>

#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <PVLayerStackModel.h>
#include <PVLayerStackWidget.h>


#include <vector>

namespace PVInspector {

typedef std::vector<int> MatchingTable_t;

class PVMainWindow;
class PVListingModel;
class PVListingView;
class PVAxesCombinationDialog;
class PVExtractorWidget;
class PVViewsListingWidget;

/**
 *  \class PVTabSplitter
 */
class PVTabSplitter : public QSplitter
{
	Q_OBJECT

private:

public:
	MatchingTable_t sortMatchingTable; //!<the table sort, modify this array to order the values. sortMatchingTable[0] is the position of the line 0 after sort.
	MatchingTable_t sortMatchingTable_invert; //!<sortMatchingTable_invert[E] (E: a line in sorted table) return the real position in the nraw 
	PVMainWindow     *main_window;   //!< The parent PVMainWindow of this PVTabSplitter
	Picviz::PVSource_p _lib_src;    //!< The Picviz::PVSource object this tab is bound to

	PVListingView *pv_listing_view; //!< The PVListingView attached with our main application

	PVListingModel *pv_listing_model; //!< The listing model

	PVLayerStackModel  *pv_layer_stack_model;
	PVLayerStackWidget *pv_layer_stack_widget;

	PVViewsListingWidget* _views_widget;

	PVExtractorWidget *_pv_extractor; //!< The extractor widget of this view
	PVAxesCombinationDialog *pv_axes_combination_editor;

	int screenshot_index;

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
	PVTabSplitter(PVMainWindow *mw, Picviz::PVSource_p lib_src, QWidget *parent);

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
	 * @return a pointer to the current Picviz::PVView attached to this PVMainSplitter
	 */
	Picviz::PVView_p get_lib_view() const
	{
		Picviz::PVView_p ret(_lib_src->current_view());
		assert(ret);
		return ret;
	}

	/**
	 *
	 * @return a pointer to the bound Picviz::PVSource
	 */
	Picviz::PVSource_p get_lib_src() const { return _lib_src; }

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

	PVAxesCombinationDialog* get_axes_combination_editor() const { return pv_axes_combination_editor; }


	QString get_tab_name() { return get_tab_name(_lib_src); }
	static QString get_tab_name(Picviz::PVSource_p src) { return src->get_name() + QString(" / ") + src->get_format_name(); }
	QString get_src_name() { return _lib_src->get_name(); }
	QString get_src_type() { return _lib_src->get_format_name(); }

	/**
	 *
	 * @return the index of the next screenshot
	 */
	int get_screenshot_index();

	MatchingTable_t *getSortMatchingTable(){return &sortMatchingTable;}

	/**
	 * Increments the index of the next screenshot
	 */
	void increment_screenshot_index();

	/**
	 * Update filter menu enabling
	 */
	void updateFilterMenuEnabling();

	void select_view(Picviz::PVView_p view);


	void create_new_mapped();
	void select_plotted(Picviz::PVPlotted* plotted);
	void create_new_plotted(Picviz::PVMapped* mapped_parent);
	void edit_mapped(Picviz::PVMapped* mapped);
	void edit_plotted(Picviz::PVPlotted* plotted);

public slots:
	/**
	 * The Slot that will refresh the PVLayerStackWidget
	 */
	void refresh_layer_stack_view_Slot(); // From PVLayerStackWindow

	void refresh_axes_combination_Slot();

signals:
	/**
	 * The selection has changed
	 */
	void selection_changed_signal(bool);

public:
};
}

#endif
