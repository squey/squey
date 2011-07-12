//! \file PVView.h
//! $Id: PVView.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVVIEW_H
#define PICVIZ_PVVIEW_H

#include <QList>
#include <QStringList>
#include <QString>
#include <QVector>
#include <QMutex>

#include <pvcore/general.h>

#include <picviz/arguments.h>
//#include <picviz/eventline.h>
//#include <picviz/index-array.h>
//#include <picviz/layer.h>
//#include <picviz/layer-stack.h>
/* #include <picviz/state-machine.h> */
//#include <picviz/square-area.h>
//#include <picviz/tags.h>
//#include <picviz/z-level-array.h>

#include <picviz/PVColor.h>
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
#include <picviz/PVZLevelArray.h>


#include <pvfilter/PVArgument.h>
#include <pvrush/PVExtractor.h>

#include <boost/shared_ptr.hpp>

namespace Picviz {

/**
 * \class PVView
 */
class LibExport PVView {


public:
	typedef QHash<QString,PVFilter::PVArgumentList> map_filter_arguments;
	typedef boost::shared_ptr<PVView> p_type;
public:
	PVView(PVPlotted_p parent);
	~PVView();

	/* Variables */
	PVRoot_p   root;
	QString    name;
	int active_axis;
	PVAxesCombination axes_combination;
	/* picviz_line_properties_t default_zombie_line_properties; */
	PVColor default_zombie_line_properties;
	PVSelection floating_selection;
	PVLayer pre_filter_layer;
	PVLayer post_filter_layer;
	PVLayer layer_stack_output_layer;
	PVLayer output_layer;
	PVPlotted_p plotted;
	PVRow row_count;
	PVLayerStack layer_stack;
	PVIndexArray nu_index_array;
	PVIndexArray nz_index_array;
	PVIndexArray nznu_index_array;
	PVSelection nu_selection;
	PVSelection real_output_selection;
	PVEventline eventline;
	PVZLevelArray z_level_array;
	PVSquareArea square_area;
	Picviz::PVStateMachine *state_machine;
	//PVTags tags;
	PVSelection volatile_selection;
	map_filter_arguments filters_args;
	int last_extractor_batch_size;
    
    QMutex gl_call_locker;


	/* Functions */
	int get_axes_count();

	/**
	 * Gets the QStringList of all Axes names according to the current PVAxesCombination
	 *
	 * @return The list of all names of all current axes
	 *
	 */
	QStringList get_axes_names_list();
	
	/**
	 * Gets the name of the chosen axis according to the actual PVAxesCombination
	 *
	 * @param index The index of the axis (starts at 0)
	 *
	 * @return The name of that axis
	 *
	 */
	QString get_axis_name(PVCol index);

	Picviz::PVColor get_color_in_output_layer(PVRow index);
	PVCol get_column_count();
	float get_column_count_as_float();
	PVRoot_p get_root();
	PVSelection &get_floating_selection();
	int get_layer_index(int index);
	float get_layer_index_as_float(int index);
	PVLayerStack &get_layer_stack();
	int get_layer_stack_layer_n_locked_state(int n);
	QString get_layer_stack_layer_n_name(int n);
	int get_layer_stack_layer_n_visible_state(int n);
	PVLayer &get_layer_stack_output_layer();
	
	bool get_line_state_in_layer_stack_output_layer(PVRow index);
	bool get_line_state_in_layer_stack_output_layer(PVRow index) const;
	bool get_line_state_in_output_layer(PVRow index);
	bool get_line_state_in_output_layer(PVRow index) const;
	bool get_line_state_in_pre_filter_layer(PVRow index);
	bool get_line_state_in_pre_filter_layer(PVRow index) const;

	int get_nu_index_count();
	int get_nu_real_row_index(int index);
	PVSelection &get_nu_selection();
	int get_number_of_selected_lines();
	int get_nz_index_count();
	int get_nz_real_row_index(int index);
	int get_nznu_index_count();
	int get_nznu_real_row_index(int index);


	int get_original_axes_count();
	PVLayer &get_output_layer();

	PVRush::PVExtractor& get_extractor();


	PVLayer &get_post_filter_layer();
	PVLayer &get_pre_filter_layer();

	PVSelection &get_real_output_selection();
	int get_real_row_index(int index);
	PVRow get_row_count();

	void reset_layers();

	int move_active_axis_closest_to_position(float x);

	void refresh_nu_index_array();
	void refresh_nz_index_array();
	void refresh_nznu_index_array();


	void set_active_axis_closest_to_position(float x);
	void set_axis_name(PVCol index, const QString &name_);
	
	void set_color_on_active_layer(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
	void set_color_on_post_filter_layer(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

	int set_layer_stack_layer_n_name(int n, char *new_name);

	void set_layer_stack_selected_layer_index(int index);

	void set_floating_selection(PVSelection &selection);

	//void set_selection_with_square_area_selection(PVSelection &selection, float xmin, float ymin, float xmax, float ymax);
	void set_selection_with_final_selection(PVSelection &selection);


	int toggle_layer_stack_layer_n_locked_state(int n);
	int toggle_layer_stack_layer_n_visible_state(int n);



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

/******************************************************************************
******************************************************************************
*
* SPECIFIC functions
*
******************************************************************************
*****************************************************************************/

	QString apply_filter_from_name(char *name, PVFilter::PVArgumentList &arguments);
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

	/**
	 * Gets the data directly from nraw, without #PVAxesCombination
	 *
	 * @param row The row number
	 * @param column The column number
	 *
	 * @return a string containing wanted data
	 *
	 */
	QString get_data_raw(PVRow row, PVCol column);

	void selection_A2B_select_with_square_area(PVSelection &a, PVSelection &b);
	

/******************************************************************************
******************************************************************************
*
* ANCESTORS
*
******************************************************************************
*****************************************************************************/

	const PVMapped_p get_mapped_parent() const;
	PVMapped_p get_mapped_parent();
	
	PVRush::PVNraw::nraw_table& get_qtnraw_parent();
	const PVRush::PVNraw::nraw_table& get_qtnraw_parent() const;
	
	PVPlotted_p get_plotted_parent();
	const PVPlotted_p get_plotted_parent() const;
	
	PVSource_p get_source_parent();
	const PVSource_p get_source_parent() const;

	void debug();

	bool is_consistent() const ;
	void set_consistent(bool c);

	void recreate_mapping_plotting();

protected:
	bool _is_consistent;
    


};

typedef PVView::p_type PVView_p;

}

#endif	/* PICVIZ_PVVIEW_H */
