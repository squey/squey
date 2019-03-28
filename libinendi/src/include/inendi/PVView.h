/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVVIEW_H
#define INENDI_PVVIEW_H

#include <inendi/PVAxesCombination.h>
#include <inendi/PVLayer.h>
#include <inendi/PVLayerStack.h>
#include <inendi/PVLinesProperties.h>
#include <inendi/PVStateMachine.h>

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVDataTreeObject.h>

#include <pvbase/types.h> // for PVCol, PVRow

#include <sigc++/sigc++.h>

#include <QStringList>
#include <QString>
#include <QColor>
#include <QHash>

#include <cstddef> // for size_t
#include <cstdint> // for int32_t
#include <string>  // for allocator, string, etc
#include <vector>  // for vector

namespace Inendi
{
class PVPlotted;
} // namespace Inendi
namespace Inendi
{
class PVSelection;
} // namespace Inendi
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore
namespace PVRush
{
class PVAxisFormat;
} // namespace PVRush
namespace PVRush
{
class PVNraw;
} // namespace PVRush
namespace pvcop
{
namespace db
{
class indexes;
} // namespace db
} // namespace pvcop
namespace tbb
{
class task_group_context;
} // namespace tbb

namespace Inendi
{
/**
 * \class PVView
 */
class PVView : public PVCore::PVDataTreeChild<PVPlotted, PVView>
{
  public:
	typedef QHash<QString, PVCore::PVArgumentList> map_filter_arguments;
	typedef int32_t id_t;

  public:
	explicit PVView(PVPlotted& plotted);
	~PVView();

  public:
	std::string get_serialize_description() const override { return "View: " + get_name(); }

	/* Functions */
	/**
	 * Gets the QStringList of all Axes names according to the current PVAxesCombination
	 *
	 * @return The list of all names of all current axes
	 *
	 */
	QStringList get_axes_names_list() const;
	QStringList get_zones_names_list() const;

	/**
	 * Gets the name of the chosen axis according to the actual PVAxesCombination
	 *
	 * @param index The index of the axis (starts at 0)
	 *
	 * @return The name of that axis
	 *
	 */
	const QString& get_axis_name(PVCombCol index) const;
	PVRush::PVAxisFormat const& get_axis(PVCombCol const comb_index) const;
	bool is_last_axis(PVCombCol const axis_comb) const
	{
		return axis_comb == get_column_count() - 1;
	}

	const PVCore::PVHSVColor get_color_in_output_layer(PVRow index) const;
	PVCombCol get_column_count() const;
	PVLayerStack& get_layer_stack();
	inline PVLayerStack const& get_layer_stack() const { return layer_stack; };

	QString get_layer_stack_layer_n_name(int n) const;
	int get_layer_stack_layer_n_visible_state(int n) const;
	PVLayer& get_layer_stack_output_layer();
	PVLayer const& get_layer_stack_output_layer() const { return layer_stack_output_layer; }
	void hide_layers()
	{
		_layer_stack_about_to_refresh.emit();
		layer_stack.hide_layers();
		_layer_stack_refreshed.emit();
	}

	PVAxesCombination const& get_axes_combination() const { return _axes_combination; }
	void set_axes_combination(std::vector<PVCol> const& comb);

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

	inline id_t get_display_view_id() const { return _view_id + 1; }

	QString get_nraw_axis_name(PVCol axis_id) const;

	PVLayer const& get_output_layer() const { return output_layer; }

	std::string get_name() const;
	QString get_window_name() const;

	QColor get_color() const { return _color; }

	PVLayer& get_post_filter_layer();

	PVSelection const& get_real_output_selection() const;

	PVRow get_row_count() const;

	void set_color_on_active_layer(const PVCore::PVHSVColor c);

	void set_layer_stack_layer_n_name(int n, QString const& name);

	void set_layer_stack_selected_layer_index(int index);

	void set_selection_from_layer(PVLayer const& layer);

	/**
	 * Set the current selected events set
	 *
	 * @param sel the new selection
	 * @param update_ls a flag to tell to update the layer-stack or the post filter layer
	 */
	void set_selection_view(PVSelection const& sel, bool update_ls = false);

	void toggle_layer_stack_layer_n_visible_state(int n);
	void move_selected_layer_to(int new_index);

	void select_all();
	void select_none();
	void select_inverse();

	void toggle_listing_unselected_visibility();
	void toggle_listing_zombie_visibility();
	void toggle_view_unselected_zombie_visibility();
	bool& are_view_unselected_zombie_visible();

	void compute_layer_min_max(Inendi::PVLayer& layer);
	void update_current_layer_min_max();
	void compute_selectable_count(Inendi::PVLayer& layer);

	void recompute_all_selectable_count();

	/**
	 * Save added dataset to mineset to remove them at the end of the inspection.
	 */
	void add_mineset_dataset(const std::string& dataset_url)
	{
		_mineset_datasets.emplace_back(dataset_url);
	}

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

	void process_correlation();

	/**
	 * Compute a merge of all visibles layer of the layer stack into layer_stack_output_layerq
	 *
	 * @param emit_signal a flag to notify listeners that a change occurs or not
	 */
	void process_layer_stack(bool emit_signal = true);

	/**
	 * Recompute post_filter_layer selection
	 *
	 * @param emit_signal a flag to notify listeners that a change occurs or not
	 *
	 * @todo this method always copies the lines properties while it should do it only once
	 */
	void process_post_filter_layer(bool emit_signal = true);

	/**
	 * Compute output layer from post_filter_layer data.
	 *
	 * @param emit_signal a flag to notify listeners that a change occurs or not
	 */
	void process_output_layer(bool emit_signal = true);

	/******************************************************************************
	******************************************************************************
	*
	* SPECIFIC functions
	*
	******************************************************************************
	*****************************************************************************/

	/**
	 * Gets the data using #PVAxesCombination
	 *
	 * @param row The row number
	 * @param column The column number
	 *
	 * @return a string containing wanted data
	 *
	 */
	std::string get_data(PVRow row, PVCombCol column) const;

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
	void sort_indexes(PVCol col,
	                  pvcop::db::indexes& idxes,
	                  tbb::task_group_context* ctxt = nullptr) const;

	/******************************************************************************
	******************************************************************************
	*
	* ANCESTORS
	*
	******************************************************************************
	*****************************************************************************/

	PVRush::PVNraw& get_rushnraw_parent();
	PVRush::PVNraw const& get_rushnraw_parent() const;

	PVCol get_nraw_axis_index(PVCombCol col) const;

	PVRow get_plotted_col_min_row(PVCombCol const combined_col) const;
	PVRow get_plotted_col_max_row(PVCombCol const combined_col) const;

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static Inendi::PVView& serialize_read(PVCore::PVSerializeObject& so, Inendi::PVPlotted& parent);

  public:
	// axis <-> section synchronisation
	void set_axis_hovered(PVCombCol col, bool entered) { _axis_hovered.emit(col, entered); }
	void set_axis_clicked(PVCombCol col) const { _axis_clicked.emit(col); }

	void set_section_clicked(PVCombCol col, size_t pos) const { _section_clicked.emit(col, pos); }

	sigc::signal<void, PVCombCol, bool> _axis_hovered;
	sigc::signal<void, PVCombCol> _axis_clicked;
	sigc::signal<void, PVCombCol, size_t> _section_clicked;
	sigc::signal<void> _axis_combination_updated;
	sigc::signal<void> _axis_combination_about_to_update;
	sigc::signal<void> _update_current_min_max;
	sigc::signal<void> _layer_stack_about_to_refresh;
	sigc::signal<void> _layer_stack_refreshed;
	sigc::signal<void> _toggle_unselected_zombie_visibility;
	sigc::signal<void> _update_layer_stack_output_layer;
	sigc::signal<void> _update_output_selection;
	sigc::signal<void> _update_output_layer;
	sigc::signal<void> _toggle_unselected;
	sigc::signal<void> _toggle_zombie;
	sigc::signal<void> _about_to_be_delete;

  protected:
	PVSelection _view_selection; //!< pre layer-stack masking selection
	PVLayer
	    post_filter_layer; //!< Contains selection and color lines for in progress view computation.
	PVLayer layer_stack_output_layer; //!< Layer grouping every information from the layer stack
	PVLayer output_layer;             //!< This is the shown layer.
	PVLayerStack layer_stack;
	PVStateMachine _state_machine;

	/*! \brief PVView's specific axes combination
	 *  It is originaly copied from the parent's PVSource, and then become specific
	 *  to that view.
	 */
	PVAxesCombination _axes_combination;

	QString _last_filter_name;
	map_filter_arguments filters_args;
	id_t _view_id;
	PVCol _active_axis;
	QColor _color;

	std::vector<std::string> _mineset_datasets; //!< Names of the exported dataset.

  private:
	static PVCore::PVHSVColor _default_zombie_line_properties; //!< Default color for Zombies lines.
};
} // namespace Inendi

#endif /* INENDI_PVVIEW_H */
