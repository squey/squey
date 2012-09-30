/**
 * \file PVTabSplitter.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVTABSPLITTER_H
#define PVTABSPLITTER_H


#include <QSplitter>

#include <pvkernel/rush/PVControllerJob.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvhive/PVObserverSignal.h>

#include <vector>

namespace PVGuiQt {
class PVAxesCombinationDialog;

class PVLayerStackWidget;

class PVListingModel;
class PVListingSortFilterProxyModel;
class PVListingView;

class PVRootTreeModel;
class PVRootTreeView;

class PVListDisplayDlg;
}

namespace PVInspector {

typedef std::vector<int> MatchingTable_t;

class PVAxisPropertiesWidget;
class PVMainWindow;
class PVExtractorWidget;

/**
 *  \class PVTabSplitter
 */
class PVTabSplitter : public QSplitter
{
	Q_OBJECT

private:
	class PVViewWidgets
	{
		friend class PVTabSplitter;
	public:
		PVViewWidgets(Picviz::PVView* view, PVTabSplitter* tab);
		PVViewWidgets() { pv_axes_combination_editor = NULL; pv_axes_properties = NULL; }
		~PVViewWidgets();
	protected:
		void delete_widgets();
	public:
		PVGuiQt::PVAxesCombinationDialog *pv_axes_combination_editor;
		PVAxisPropertiesWidget  *pv_axes_properties;
	};

	friend class PVViewWidgets;

public:

public:
	/**
	 * Constructor.
	 *
	 * @param mw
	 * @param lib_view
	 * @param parent
	 */
	PVTabSplitter(Picviz::PVSource& lib_src, QWidget *parent);

	virtual ~PVTabSplitter();

	/**
	 *
	 * @return a pointer to the PVLayerStackWidget attached to this PVMainSplitter
	 */
	PVGuiQt::PVLayerStackWidget *get_layer_stack_widget() const { return pv_layer_stack_widget; }

	PVGuiQt::PVListingSortFilterProxyModel* get_listing_proxy_model() { return pv_listing_proxy_model; }
	PVGuiQt::PVListingView* get_listing_view() { return pv_listing_view; }


	PVGuiQt::PVListDisplayDlg* get_source_invalid_elts_dlg() { return _inv_elts_dlg; };

	/**
	 *
	 * @return a pointer to the current Picviz::PVView attached to this PVMainSplitter
	 */
	Picviz::PVView* get_lib_view()
	{
		Picviz::PVView* ret(get_lib_src()->current_view());
		assert(ret);
		return ret;
	}

	/**
	 *
	 * @return a pointer to the bound Picviz::PVSource
	 */
	Picviz::PVSource* get_lib_src() { return _obs_src.get_object(); }

	void ensure_column_visible(PVCol col);

	/**
	 *
	 * @return a pointer to the PVInspector::PVExtractorWidget attached to this PVMainSplitter
	 */
	PVExtractorWidget* get_extractor_widget() const {return _pv_extractor;}

	PVGuiQt::PVAxesCombinationDialog* get_axes_combination_editor(Picviz::PVView* view);

	PVAxisPropertiesWidget* get_axes_properties_widget(Picviz::PVView* view);

	QString get_current_view_name() { return get_current_view_name(get_lib_src()); };
	static QString get_current_view_name(Picviz::PVSource* src);
	QString get_tab_name() { return get_tab_name(get_lib_src()); }
	static QString get_tab_name(Picviz::PVSource* src) { return src->get_window_name(); }
	QString get_src_name() { return get_lib_src()->get_name(); }
	QString get_src_type() { return get_lib_src()->get_format_name(); }

	PVViewWidgets const& get_view_widgets(Picviz::PVView* view);

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

	void select_view(Picviz::PVView* view);


	void create_new_mapped();
	void select_plotted(Picviz::PVPlotted* plotted);
	void create_new_plotted(Picviz::PVMapped* mapped_parent);
	void edit_mapped(Picviz::PVMapped* mapped);
	void process_mapped_if_current(Picviz::PVMapped* mapped);
	void edit_plotted(Picviz::PVPlotted* plotted);
	void process_plotted_if_current(Picviz::PVPlotted* plotted);
	void toggle_listing_sort();

	size_t get_unique_indexes_for_current_listing(PVCol column, std::vector<int>& idxes);

	void emit_source_changed() { emit source_changed(); }

public:
	bool process_extraction_job(PVRush::PVControllerJob_p job);

private slots:
	void source_about_to_be_deleted();

signals:
	/**
	 * The selection has changed
	 */
	void selection_changed_signal(bool);

	/*
	 * The source has changed
	 */
	void source_changed();

private:
	PVGuiQt::PVListingView *pv_listing_view; //!< The PVListingView attached with our main application

	PVGuiQt::PVListingModel *pv_listing_model; //!< The listing model
	PVGuiQt::PVListingSortFilterProxyModel* pv_listing_proxy_model;

	PVGuiQt::PVLayerStackWidget *pv_layer_stack_widget;

	PVGuiQt::PVRootTreeView*  _data_tree_view;
	PVGuiQt::PVRootTreeModel* _data_tree_model;

	PVGuiQt::PVListDisplayDlg* _inv_elts_dlg;

	PVExtractorWidget *_pv_extractor; //!< The extractor widget of this view

	int screenshot_index;

	QHash<Picviz::PVView const*, PVViewWidgets> _view_widgets;

	PVHive::PVObserverSignal<Picviz::PVSource> _obs_src;
};

}

#endif
