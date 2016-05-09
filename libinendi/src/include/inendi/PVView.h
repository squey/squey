/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVVIEW_H
#define INENDI_PVVIEW_H

#include <QList>
#include <QStringList>
#include <QString>
#include <QVector>
#include <QMutex>

#include <pvcop/db/array.h>

#include <pvkernel/core/general.h>

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVLinesProperties.h>
#include <inendi/PVAxesCombination.h>
#include <inendi/PVLayerStack.h>
#include <inendi/PVStateMachine.h>

#include <inendi/PVView_types.h>

namespace Inendi
{

/**
 * \class PVView
 */
typedef typename PVCore::PVDataTreeObject<PVPlotted, PVCore::PVDataTreeNoChildren<PVView>>
    data_tree_view_t;
class PVView : public data_tree_view_t
{
	friend class PVCore::PVSerializeObject;
	friend class PVRoot;
	friend class PVScene;
	friend class PVSource;

  public:
	typedef QHash<QString, PVCore::PVArgumentList> map_filter_arguments;
	typedef int32_t id_t;
	typedef PVAxesCombination::axes_comb_id_t axes_comb_id_t;

  public:
	PVView(PVPlotted* plotted);
	PVView(const PVView& org) = delete;
	~PVView();

  protected:
	// For PVSource
	inline void set_view_id(id_t id) { _view_id = id; }

  public:
	inline PVSelection& get_floating_selection() { return floating_selection; }
	inline PVSelection& get_volatile_selection() { return volatile_selection; }

	// Proxy functions for PVHive
	void remove_column(PVCol index) { _axes_combination.remove_axis(index); }
	bool move_axis_to_new_position(PVCol index_source, PVCol index_dest)
	{
		return _axes_combination.move_axis_to_new_position(index_source, index_dest);
	}
	void axis_append(const PVAxis& axis) { _axes_combination.axis_append(axis); }

	virtual QString get_serialize_description() const { return "View: " + get_name(); }

	/* Functions */
	PVCol get_axes_count() const;

	/**
	 * Gets the QStringList of all Axes names according to the current PVAxesCombination
	 *
	 * @return The list of all names of all current axes
	 *
	 */
	QStringList get_axes_names_list() const;
	QStringList get_zones_names_list() const;
	inline QStringList get_original_axes_names_list() const
	{
		return get_axes_combination().get_original_axes_names_list();
	}

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
	PVAxis const& get_axis(PVCol const comb_index) const;
	PVAxis const& get_axis_by_id(axes_comb_id_t const axes_comb_id) const;
	bool is_last_axis(axes_comb_id_t const axes_comb_id) const
	{
		return get_axes_combination().is_last_axis(axes_comb_id);
	}
	bool is_last_axis(PVCol const axis_comb) const { return axis_comb == get_column_count() - 1; }

	const PVCore::PVHSVColor get_color_in_output_layer(PVRow index) const;
	PVCol get_column_count() const;
	PVLayerStack& get_layer_stack();
	inline PVLayerStack const& get_layer_stack() const { return layer_stack; };
	int get_layer_stack_layer_n_locked_state(int n) const;
	;
	QString get_layer_stack_layer_n_name(int n) const;
	int get_layer_stack_layer_n_visible_state(int n) const;
	PVLayer& get_layer_stack_output_layer();
	PVLayer const& get_layer_stack_output_layer() const { return layer_stack_output_layer; }
	void hide_layers() { layer_stack.hide_layers(); }

	PVCol get_active_axis() const
	{
		assert(_active_axis < get_column_count());
		return _active_axis;
	}
	PVStateMachine& get_state_machine() { return _state_machine; }
	PVStateMachine const& get_state_machine() const { return _state_machine; }

	PVAxesCombination const& get_axes_combination() const { return _axes_combination; }
	void set_axes_combination_list_id(PVAxesCombination::columns_indexes_t const& idxes,
	                                  PVAxesCombination::list_axes_t const& axes);

	inline PVLayer const& get_current_layer() const { return layer_stack.get_selected_layer(); }
	inline PVLayer& get_current_layer() { return layer_stack.get_selected_layer(); }

	inline void move_selected_layer_up() { layer_stack.move_selected_layer_up(); }
	inline void move_selected_layer_down() { layer_stack.move_selected_layer_down(); }

	inline PVCore::PVHSVColor const* get_output_layer_color_buffer() const
	{
		return output_layer.get_lines_properties().get_buffer();
	}

	bool get_line_state_in_layer_stack_output_layer(PVRow index) const;
	bool get_line_state_in_output_layer(PVRow index) const;
	PVSelection const& get_selection_visible_listing() const;

	int get_number_of_selected_lines() const;

	inline id_t get_view_id() const { return _view_id; }
	inline id_t get_display_view_id() const { return _view_id + 1; }

	PVCol get_original_axes_count() const;
	QString get_original_axis_name(PVCol axis_id) const;
	QString get_original_axis_type(PVCol axis_id) const;
	inline PVCol get_original_axis_index(PVCol view_idx) const
	{
		return _axes_combination.get_axis_column_index(view_idx);
	}

	PVLayer const& get_output_layer() const { return output_layer; }

	QString get_name() const;
	QString get_window_name() const;

	void set_color(QColor color) { _color = color; }
	QColor get_color() const { return _color; }

	PVLayer& get_post_filter_layer();

	PVSelection const& get_real_output_selection() const;

	PVRow get_row_count() const;
	void set_row_count(PVRow row_count);

	void reset_layers();

	int move_active_axis_closest_to_position(float x);
	PVCol get_active_axis_closest_to_position(float x);

	void set_active_axis_closest_to_position(float x);
	void set_axis_name(PVCol index, const QString& name_);

	void set_color_on_active_layer(const PVCore::PVHSVColor c);

	int set_layer_stack_layer_n_name(int n, QString const& name);

	void set_layer_stack_selected_layer_index(int index);

	void set_floating_selection(PVSelection& selection);

	void set_selection_from_layer(PVLayer const& layer);
	void set_selection_view(PVSelection const& sel);

	int toggle_layer_stack_layer_n_locked_state(int n);
	int toggle_layer_stack_layer_n_visible_state(int n);
	void move_selected_layer_to(int new_index);

	void select_all_nonzb_lines();
	void select_no_line();
	void select_inv_lines();

	void toggle_listing_unselected_visibility();
	void toggle_listing_zombie_visibility();
	void toggle_view_unselected_zombie_visibility();
	bool& are_view_unselected_zombie_visible();

	void compute_layer_min_max(Inendi::PVLayer& layer);
	void compute_selectable_count(Inendi::PVLayer& layer);

	void recompute_all_selectable_count();

	/**
	 * do any process after a mapped load
	 */
	void finish_process_from_rush_pipeline();

#ifdef WITH_MINESET
	/**
	 * Save added dataset to mineset to remove them at the end of the inspection.
	 */
	void add_mineset_dataset(const std::string& dataset_url)
	{
		_mineset_datasets.emplace_back(dataset_url);
	}
#endif

	/******************************************************************************
	******************************************************************************
	*
	* functions to manipulate the Layers involved in the View
	*
	******************************************************************************
	*****************************************************************************/

	void add_new_layer(QString name = QString());
	void delete_layer_n(int idx);
	void delete_selected_layer();
	void duplicate_selected_layer(const QString& name);
	void commit_selection_to_layer(PVLayer& layer);

	void process_from_eventline();
	void process_from_layer_stack();
	void process_from_selection();
	void process_real_output_selection();

	/**
	 * Compute a merge of all visibles layer of the layer stack.
	 *
	 * * Save data in layer_stack_output_layer.
	 */
	void process_layer_stack();

	/**
	 * Set correct selection to post_filter_layer.
	 *
	 * * Copy color (FIXME : done every time, should be done only once).
	 * * Merge selection with layer_stack selection.
	 */
	void process_selection();

	/**
	 * compute real selection (selected elements) and nu selection and copy lines properties.
	 *
	 * FIXME : Why not setting real selection without copy?
	 */
	void process_eventline();

	/**
	 * Compute selection of visible elements.
	 */
	void process_visibility();

	void process_parent_plotted();

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
	std::string get_data(PVRow row, PVCol column) const;

	/**
	 * Gets the data directly from nraw, without #PVAxesCombination
	 *
	 * @param row The row number
	 * @param column The column number
	 *
	 * @return a string containing wanted data
	 *
	 */
	std::string get_data_raw(PVRow row, PVCol column) const
	{
		return get_rushnraw_parent().at_string(row, column);
	}

	void commit_volatile_in_floating_selection();

	/***********
	 * FILTERS
	 ***********/
	inline QString const& get_last_used_filter() const { return _last_filter_name; }
	inline void set_last_used_filter(QString const& name) { _last_filter_name = name; }
	inline bool is_last_filter_used_valid() const { return !_last_filter_name.isEmpty(); }
	inline PVCore::PVArgumentList& get_last_args_filter(QString const& name)
	{
		return filters_args[name];
	}

	/**
	 * Sorting functions
	 *
	 * It sorts idxes based on "col" values.
	 * "col" is the column id without axis combination modification.
	 */
	void
	sort_indexes(PVCol col, pvcop::db::indexes& idxes, tbb::task_group_context* ctxt = NULL) const;

	/******************************************************************************
	******************************************************************************
	*
	* ANCESTORS
	*
	******************************************************************************
	*****************************************************************************/

	PVRush::PVNraw& get_rushnraw_parent()
	{
		assert(_rushnraw_parent);
		return *_rushnraw_parent;
	};
	PVRush::PVNraw const& get_rushnraw_parent() const
	{
		assert(_rushnraw_parent);
		return *_rushnraw_parent;
	};

	PVCol get_real_axis_index(PVCol col) const;

	PVRow get_plotted_col_min_row(PVCol const combined_col) const;
	PVRow get_plotted_col_max_row(PVCol const combined_col) const;

  public:
	// State machine
	inline void set_square_area_mode(PVStateMachine::SquareAreaModes mode)
	{
		_state_machine.set_square_area_mode(mode);
	}

  protected:
	/******************************************************************************
	******************************************************************************
	*
	* Serialization
	*
	******************************************************************************
	*****************************************************************************/
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);

  public:
	PVSERIALIZEOBJECT_SPLIT

  protected:
	/*! \brief PVView's specific axes combination
	 *  It is originaly copied from the parent's PVSource, and then become specific
	 *  to that view.
	 */
	PVAxesCombination _axes_combination;

	PVSelection floating_selection;   //!< This is the current selection
	PVLayer post_filter_layer;        //!< This is the result of the filtering. TODO : FIXME
	PVLayer layer_stack_output_layer; //!< Layer grouping every information from the layer stack
	PVLayer output_layer;             //!< This is the shown layer.
	PVLayerStack layer_stack;
	PVSelection real_output_selection; //!< This is selected elements
	PVStateMachine _state_machine;
	PVSelection volatile_selection; //!< It is the selection currently computed. It will be flush in
	// floating_selection once it is completed.

	QString _last_filter_name;
	map_filter_arguments filters_args;
	PVRush::PVNraw* _rushnraw_parent = nullptr; //!< Pointer to the NRaw from source.
	id_t _view_id;
	PVCol _active_axis;
	QColor _color;

#ifdef WITH_MINESET
	std::vector<std::string> _mineset_datasets; //!< Names of the exported dataset.
#endif

  private:
	static PVCore::PVHSVColor _default_zombie_line_properties; //!< Default color for Zombies lines.
};

typedef PVView::p_type PVView_p;
}

#endif /* INENDI_PVVIEW_H */
