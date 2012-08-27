/**
 * \file PVView.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVVIEW_H
#define PICVIZ_PVVIEW_H

#include <QList>
#include <QStringList>
#include <QString>
#include <QVector>
#include <QMutex>

#include <pvkernel/core/general.h>

#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/rush/PVExtractor.h>

#include <picviz/PVLinesProperties.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVRoot.h>
#include <picviz/PVSource.h>
#include <picviz/PVEventline.h>
#include <picviz/PVLayerStack.h>
#include <picviz/PVIndexArray.h>
#include <picviz/PVSquareArea.h>
#include <picviz/PVStateMachine.h>
#include <picviz/PVSortingFunc.h>
#include <picviz/PVDefaultSortingFunc.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVZLevelArray.h>

#include <picviz/PVView_types.h>
#include <picviz/PVView_impl.h>

#include <boost/enable_shared_from_this.hpp>

namespace Picviz {

/**
 * \class PVView
 */
typedef typename PVCore::PVDataTreeObject<PVPlotted, PVCore::PVDataTreeNoChildren<PVView> > data_tree_view_t;
class LibPicvizDecl PVView: public data_tree_view_t
{
	friend class PVCore::PVSerializeObject;
	friend class PVSource;
	friend class PVScene;
	friend class PVCore::PVDataTreeAutoShared<PVView>;
public:
	typedef QHash<QString,PVCore::PVArgumentList> map_filter_arguments;
	typedef int32_t id_t;
	
protected:
	PVView();

public:
	virtual ~PVView();

protected:
	PVView(const PVView& org);

	// For PVSource
	void add_column(PVAxis const& axis);
	inline void set_view_id(id_t id) { _view_id = id; }

public:

	/* Variables */
	QString    name;
	int active_axis;

	/*! \brief PVView's specific axes combination
	 *  It is originaly copied from the parent's PVSource, and then become specific
	 *  to that view.
	 */
	PVAxesCombination axes_combination;

	PVCore::PVColor default_zombie_line_properties;
	PVSelection floating_selection;
	PVLayer pre_filter_layer;
	PVLayer post_filter_layer;
	PVLayer layer_stack_output_layer;
	PVLayer output_layer;
	PVRow row_count;
	PVLayerStack layer_stack;
	PVSelection nu_selection;
	PVSelection real_output_selection;
	PVEventline eventline;
	PVZLevelArray z_level_array;
	PVSquareArea square_area;
	Picviz::PVStateMachine *state_machine;
	PVSelection volatile_selection;
	int last_extractor_batch_size;
    
    QMutex gl_call_locker;

    // Proxy functions for PVHive
	void remove_column(PVCol index) { axes_combination.remove_axis(index); }
	bool move_axis_to_new_position(PVCol index_source, PVCol index_dest) { return axes_combination.move_axis_to_new_position(index_source, index_dest); }
	void axis_append(const PVAxis &axis) { axes_combination.axis_append(axis); }

	//void init_from_plotted(PVPlotted* parent, bool keep_layers);

	virtual QString get_serialize_description() const { return "View: " + get_name(); }

	/* Functions */
	PVCol get_axes_count() const;

	template <class T>
	QList<PVCol> get_original_axes_index_with_tag(T const& tag) const
	{
		return axes_combination.get_original_axes_index_with_tag<T>(tag);
	}

	/**
	 * Gets the QStringList of all Axes names according to the current PVAxesCombination
	 *
	 * @return The list of all names of all current axes
	 *
	 */
	QStringList get_axes_names_list() const;
	
	/**
	 * Gets the name of the chosen axis according to the actual PVAxesCombination
	 *
	 * @param index The index of the axis (starts at 0)
	 *
	 * @return The name of that axis
	 *
	 */
	const QString& get_axis_name(PVCol index) const;
	QString get_axis_type(PVCol index) const;

	void emit_user_modified_sel(QList<Picviz::PVView*>* changed_views = NULL);

	PVCore::PVColor get_color_in_output_layer(PVRow index);
	PVCol get_column_count() const;
	float get_column_count_as_float();
	PVSelection &get_floating_selection();
	int get_layer_index(int index);
	float get_layer_index_as_float(int index);
	PVLayerStack &get_layer_stack();
	inline PVLayerStack const& get_layer_stack() const { return layer_stack; };
	int get_layer_stack_layer_n_locked_state(int n);
	QString get_layer_stack_layer_n_name(int n);
	int get_layer_stack_layer_n_visible_state(int n);
	PVLayer &get_layer_stack_output_layer();

	PVAxesCombination const& get_axes_combination() const { return axes_combination; }

	inline PVLayer const& get_current_layer() const { return layer_stack.get_selected_layer(); }
	inline PVLayer& get_current_layer() { return layer_stack.get_selected_layer(); }
	
	bool get_line_state_in_layer_stack_output_layer(PVRow index);
	bool get_line_state_in_layer_stack_output_layer(PVRow index) const;
	bool get_line_state_in_output_layer(PVRow index);
	bool get_line_state_in_output_layer(PVRow index) const;
	bool get_line_state_in_pre_filter_layer(PVRow index);
	bool get_line_state_in_pre_filter_layer(PVRow index) const;
	bool is_line_visible_listing(PVRow index) const;
	bool is_real_output_selection_empty() const;
	PVSelection const* get_selection_visible_listing() const;

	PVSelection &get_nu_selection();
	int get_number_of_selected_lines();

	inline id_t get_view_id() const { return _view_id; }
	inline id_t get_display_view_id() const { return _view_id+1; }


	int get_original_axes_count();
	QString get_original_axis_name(PVCol axis_id) const;
	QString get_original_axis_type(PVCol axis_id) const;
	inline PVCol get_original_axis_index(PVCol view_idx) const { return axes_combination.get_axis_column_index(view_idx); }

	PVLayer &get_output_layer();

	PVRush::PVExtractor& get_extractor();

	QString get_name() const;
	QString get_window_name() const;


	PVLayer &get_post_filter_layer();
	PVLayer &get_pre_filter_layer();

	PVSelection &get_real_output_selection();
	PVSelection const& get_real_output_selection() const;
	int get_real_row_index(int index);
	PVRow get_row_count() const;

	void reset_layers();

	int move_active_axis_closest_to_position(float x);
	PVCol get_active_axis_closest_to_position(float x);

	void expand_selection_on_axis(PVCol axis_id, QString const& mode);

	void set_active_axis_closest_to_position(float x);
	void set_axis_name(PVCol index, const QString &name_);
	
	void set_color_on_active_layer(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
	void set_color_on_post_filter_layer(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

	int set_layer_stack_layer_n_name(int n, char *new_name);

	void set_layer_stack_selected_layer_index(int index);

	void set_floating_selection(PVSelection &selection);

	//void set_selection_with_square_area_selection(PVSelection &selection, float xmin, float ymin, float xmax, float ymax);
	void set_selection_with_final_selection(PVSelection &selection);
	void set_selection_from_layer(PVLayer const& layer);
	void set_selection_view(PVSelection const& sel);

	int toggle_layer_stack_layer_n_locked_state(int n);
	int toggle_layer_stack_layer_n_visible_state(int n);

	void select_all_nonzb_lines();
	void select_no_line();
	void select_inv_lines();

	PVSortingFunc_p get_sort_plugin_for_col(PVCol col) const;


/******************************************************************************
******************************************************************************
*
* functions to manipulate the Layers involved in the View
*
******************************************************************************
*****************************************************************************/

	int add_new_layer();
	void commit_to_new_layer();

	void load_post_to_pre();

	void process_eventline();
	void process_filter();
	void process_layer_stack();
	void process_selection();
	void process_visibility();

	void process_from_eventline();
	void process_from_filter();
	void process_from_layer_stack();
	void process_from_selection();

	void process_parent_plotted();
	void reset_view();

/******************************************************************************
******************************************************************************
*
* SPECIFIC functions
*
******************************************************************************
*****************************************************************************/

	void apply_filter_named_select_all();

	/**
	 * Gets the data using #PVAxesCombination
	 *
	 * @param row The row number
	 * @param column The column number
	 *
	 * @return a string containing wanted data
	 *
	 */
	QString get_data(PVRow row, PVCol column);
	PVCore::PVUnicodeString const& get_data_unistr(PVRow row, PVCol column);

	/**
	 * Gets the data directly from nraw, without #PVAxesCombination
	 *
	 * @param row The row number
	 * @param column The column number
	 *
	 * @return a string containing wanted data
	 *
	 */
	QString get_data_raw(PVRow row, PVCol column) const { return get_qtnraw_parent().at(row, column).get_qstr(); }
	inline PVCore::PVUnicodeString const& get_data_unistr_raw(PVRow row, PVCol column) const { return get_rushnraw_parent().at_unistr(row, column); }


	void selection_A2B_select_with_square_area(PVSelection &a, PVSelection &b);

	void commit_volatile_in_floating_selection();
	
	/***********
	 * FILTERS
	 ***********/
	inline QString const& get_last_used_filter() const { return _last_filter_name; }
	inline void set_last_used_filter(QString const& name) { _last_filter_name = name; }
	inline bool is_last_filter_used_valid() const { return !_last_filter_name.isEmpty(); }
	inline PVCore::PVArgumentList& get_last_args_filter(QString const& name) { return filters_args[name]; }


	/* Sorting and unique functions */

	// L must be a vector of integers
	template <class L>
	void sort_indexes(PVCol column, Qt::SortOrder order, L& idxes)
	{
		PVSortingFunc_p sp = get_sort_plugin_for_col(column);
		__impl::stable_sort_indexes_f(&get_rushnraw_parent(), column, sp->f(), order, idxes);
	}

	// L must be a vector of integers
	template <class L>
	void unique_indexes_copy(PVCol column, L const& idxes_in, L& idxes_out)
	{
		PVSortingFunc_p sp = get_sort_plugin_for_col(column);
		__impl::unique_indexes_copy_f<L>(&get_rushnraw_parent(), column, sp->f_equals(), idxes_in, idxes_out);
	}

	template <class L>
	size_t sort_unique_indexes(PVCol column, L& idxes)
	{
		PVSortingFunc_p sp = get_sort_plugin_for_col(column);
		__impl::sort_indexes_f(&get_rushnraw_parent(), column, sp->f_less(), Qt::AscendingOrder, idxes);
		typename L::iterator it_end = __impl::unique_indexes_f<L>(&get_rushnraw_parent(), column, sp->f_equals(), idxes);
		return it_end-idxes.begin();
	}

	// Helper functions for sorting
	template <class L>
	inline void sort_indexes_with_axes_combination(PVCol column, Qt::SortOrder order, L& idxes)
	{
		sort_indexes<L>(axes_combination.get_axis_column_index(column), order, idxes);
	}
	template <class L>
	inline void unique_indexes_copy_with_axes_combination(PVCol column, L const& idxes_in, L& idxes_out)
	{
		unique_indexes_copy<L>(axes_combination.get_axis_column_index(column), idxes_in, idxes_out);
	}
	template <class L>
	inline size_t sort_unique_indexes_with_axes_combination(PVCol column, L& idxes)
	{
		return sort_unique_indexes<L>(axes_combination.get_axis_column_index(column), idxes);
	}

	boost::weak_ptr<PVCore::PVSerializeObject> get_last_so() const { return _last_so; }
	void set_last_so(PVCore::PVSerializeObject_p const& so) { _last_so = boost::weak_ptr<PVCore::PVSerializeObject>(so); }


/******************************************************************************
******************************************************************************
*
* ANCESTORS
*
******************************************************************************
*****************************************************************************/
	
	PVRush::PVNraw::nraw_table& get_qtnraw_parent();
	const PVRush::PVNraw::nraw_table& get_qtnraw_parent() const;

	PVRush::PVNraw& get_rushnraw_parent() { assert(_rushnraw_parent); return *_rushnraw_parent; };
	PVRush::PVNraw const& get_rushnraw_parent() const { assert(_rushnraw_parent); return *_rushnraw_parent; };

	void debug();

	bool is_consistent() const ;
	void set_consistent(bool c);

	void recreate_mapping_plotting();

	PVCol get_real_axis_index(PVCol col);

protected:
	void set_parent_from_ptr(PVPlotted* plotted);
/******************************************************************************
******************************************************************************
*
* Serialization
*
******************************************************************************
*****************************************************************************/
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

/******************************************************************************
******************************************************************************
*
* Initialisation
*
******************************************************************************
*****************************************************************************/
	void init_defaults();


protected:
	bool _is_consistent;
	QString _last_filter_name;
	map_filter_arguments filters_args;
	PVRush::PVNraw* _rushnraw_parent;
	boost::weak_ptr<PVCore::PVSerializeObject> _last_so;
	id_t _view_id;
};

typedef PVView::p_type PVView_p;

}

#endif	/* PICVIZ_PVVIEW_H */
