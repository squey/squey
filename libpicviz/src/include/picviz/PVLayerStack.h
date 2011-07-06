//! \file PVLayerStack.h
//! $Id: PVLayerStack.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERSTACK_H
#define PICVIZ_PVLAYERSTACK_H

#include <pvcore/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerIndexArray.h>

#define PICVIZ_LAYER_STACK_MAX_DEPTH 256

namespace Picviz {

/**
 * \class PVLayerStack
 */
class LibExport PVLayerStack {
private:
	PVLayerIndexArray lia;
	int               layer_count; // layer_count < 256
	int               selected_layer_index;
	QList<PVLayer>    table;
public:

	/**
	 * Constructor
	 */
	PVLayerStack(PVRow row_count);

	
	int get_layer_count() const {return layer_count;}
 	PVLayer const& get_layer_n(int n) const { return table[n]; };
 	PVLayer& get_layer_n(int n) { return table[n]; };

 	PVLayer& get_selected_layer() { return table[get_selected_layer_index()]; }
 	PVLayer const& get_selected_layer() const { return table[get_selected_layer_index()]; }

	int get_selected_layer_index() const {return selected_layer_index;}
	
	void set_selected_layer_index(int index) {selected_layer_index = index;}
// 
 	void process(PVLayer &output_layer, PVRow row_count);
	void update_layer_index_array_completely();
// 
	PVLayer* append_layer(const PVLayer & layer);
	PVLayer* append_new_layer();
 	PVLayer* append_new_layer_from_selection_and_lines_properties(PVSelection const& selection, PVLinesProperties const& lines_properties);

	void delete_by_index(int index);
	void delete_all_layers();
	void delete_selected_layer();
// 
// 	picviz_layer_t *layer_get_by_index(int index);

	void move_layer_down(int index);
	void move_layer_up(int index);
	void move_selected_layer_down();
	void move_selected_layer_up();

	PVLayerIndexArray& get_lia() {return lia;}
	const PVLayerIndexArray& get_lia() const {return lia;}

	void set_row_count(PVRow row_count) { lia.set_row_count(row_count); };


// 	void debug();
};
}

#endif	/* PICVIZ_PVLAYERSTACK_H */




/*



LibExport picviz_layer_t *picviz_layer_stack_get_layer_n(picviz_layer_stack_t *layer_stack, int n);
LibExport picviz_layer_t *picviz_layer_stack_get_selected_layer(picviz_layer_stack_t *layer_stack);


LibExport void picviz_layer_stack_process(picviz_layer_stack_t *layer_stack, picviz_layer_t *output_layer, PVRow row_count);


LibExport void picviz_layer_stack_append_new_layer_from_selection_and_lines_properties(picviz_layer_stack_t *layer_stack, picviz_selection_t *selection, Picviz::PVLinesProperties *lines_properties);

LibExport void picviz_layer_stack_layer_delete_by_index(picviz_layer_stack_t *layer_stack, int index);
LibExport void picviz_layer_stack_delete_selected_layer(picviz_layer_stack_t *layer_stack);

LibExport picviz_layer_t *picviz_layer_stack_layer_get_by_index(picviz_layer_stack_t *layer_stack, int index);

*/
