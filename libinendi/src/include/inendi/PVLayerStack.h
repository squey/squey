/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERSTACK_H
#define INENDI_PVLAYERSTACK_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <inendi/PVLayer.h>
#include <inendi/PVLayerIndexArray.h>

#define INENDI_LAYER_STACK_MAX_DEPTH 256
#define INENDI_LAYERSTACK_ARCHIVE_EXT "pvls"
#define INENDI_LAYERSTACK_ARCHIVE_FILTER "INENDI layer-stack files (*." INENDI_LAYERSTACK_ARCHIVE_EXT ")"

namespace Inendi {

class PVPlotted;

/**
 * \class PVLayerStack
 */
class PVLayerStack {
	friend class PVCore::PVSerializeObject;
private:
	PVLayerIndexArray lia;
	int               layer_count; // layer_count < 256
	int               next_new_layer_counter; // counter for layers creation
	int               selected_layer_index;
	QList<PVLayer>    table;
	bool			  _should_hide_layers = true;
public:

	/**
	 * Constructor
	 */
	PVLayerStack(PVRow row_count = 0);

	QString get_new_layer_name() const;
	bool& should_hide_layers() { return _should_hide_layers; }
	int get_layer_count() const {return layer_count;}
 	PVLayer const& get_layer_n(int n) const { return table[n]; };
 	PVLayer& get_layer_n(int n) { return table[n]; };

 	PVLayer& get_selected_layer() { return table[get_selected_layer_index()]; }
 	PVLayer const& get_selected_layer() const { return table[get_selected_layer_index()]; }

	int const& get_selected_layer_index() const {return selected_layer_index;}
	
	void set_selected_layer_index(int index) {selected_layer_index = index;}
// 
 	void process(PVLayer &output_layer, PVRow row_count);
	void update_layer_index_array_completely();
// 
	PVLayer* append_layer(const PVLayer & layer);
	PVLayer* append_new_layer(PVRow row_count = INENDI_LINES_MAX, QString const& name = QString());
 	PVLayer* append_new_layer_from_selection_and_lines_properties(PVSelection const& selection, PVLinesProperties const& lines_properties);
	bool contains_layer(PVLayer* layer) const;

	void compute_min_maxs(PVPlotted const& plotted);
	void compute_selectable_count(PVRow row_count);

	void delete_by_index(int index);
	void delete_all_layers();
	void delete_selected_layer();

	PVLayer* duplicate_selected_layer(const QString &name);

	void move_layer_down(int index);
	void move_layer_up(int index);
	void move_selected_layer_to(int new_index);
	void move_selected_layer_down();
	void move_selected_layer_up();
	void hide_layers();

	PVLayerIndexArray& get_lia() {return lia;}
	const PVLayerIndexArray& get_lia() const {return lia;}

	void set_row_count(PVRow row_count) { lia.set_row_count(row_count); };

public:
	void load_from_file(QString const& path);
	void save_to_file(QString const& path);

	void copy_details_to_clipboard();

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

};
}

#endif	/* INENDI_PVLAYERSTACK_H */
