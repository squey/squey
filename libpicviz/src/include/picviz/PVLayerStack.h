/**
 * \file PVLayerStack.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERSTACK_H
#define PICVIZ_PVLAYERSTACK_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerIndexArray.h>

#define PICVIZ_LAYER_STACK_MAX_DEPTH 256
#define PICVIZ_LAYERSTACK_ARCHIVE_EXT "pvls"
#define PICVIZ_LAYERSTACK_ARCHIVE_FILTER "Picviz layer-stack files (*." PICVIZ_LAYERSTACK_ARCHIVE_EXT ")"

namespace Picviz {

class PVPlotted;

/**
 * \class PVLayerStack
 */
class LibPicvizDecl PVLayerStack {
	friend class PVCore::PVSerializeObject;
private:
	PVLayerIndexArray lia;
	int               layer_count; // layer_count < 256
	int               next_new_layer_counter; // counter for layers creation
	int               selected_layer_index;
	QList<PVLayer>    table;
public:

	/**
	 * Constructor
	 */
	PVLayerStack(PVRow row_count = 0);

	QString get_new_layer_name_from_dialog(QWidget* parent = nullptr) const;
	QString get_new_layer_name() const;
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
	PVLayer* append_new_layer(QString const& name = QString());
 	PVLayer* append_new_layer_from_selection_and_lines_properties(PVSelection const& selection, PVLinesProperties const& lines_properties);
	bool contains_layer(PVLayer* layer) const;

	void compute_min_maxs(PVPlotted const& plotted);

	void delete_by_index(int index);
	void delete_all_layers();
	void delete_selected_layer();

	void duplicate_selected_layer(const QString &name);

// 
// 	picviz_layer_t *layer_get_by_index(int index);

	void move_layer_down(int index);
	void move_layer_up(int index);
	void move_selected_layer_down();
	void move_selected_layer_up();

	PVLayerIndexArray& get_lia() {return lia;}
	const PVLayerIndexArray& get_lia() const {return lia;}

	void set_row_count(PVRow row_count) { lia.set_row_count(row_count); };

public:
	void load_from_file(QString const& path);
	void save_to_file(QString const& path);

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);


// 	void debug();
};
}

#endif	/* PICVIZ_PVLAYERSTACK_H */
