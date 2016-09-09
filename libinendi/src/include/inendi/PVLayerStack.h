/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERSTACK_H
#define INENDI_PVLAYERSTACK_H

#include <pvbase/types.h> // for PVRow

#include <QList>   // for QList
#include <QString> // for QString

namespace Inendi
{
class PVLayer;
}
namespace Inendi
{
class PVLinesProperties;
}
namespace Inendi
{
class PVSelection;
}
namespace PVCore
{
class PVSerializeObject;
}

#define INENDI_LAYER_STACK_MAX_DEPTH 256

namespace Inendi
{

/**
 * \class PVLayerStack
 */
class PVLayerStack
{
	friend class PVCore::PVSerializeObject;

  public:
	/**
	 * Constructor
	 */
	PVLayerStack();

	QString get_new_layer_name() const;
	bool& should_hide_layers() { return _should_hide_layers; }
	inline int get_layer_count() const { return _table.size(); }
	PVLayer const& get_layer_n(int n) const { return _table[n]; };
	PVLayer& get_layer_n(int n) { return _table[n]; };

	PVLayer& get_selected_layer() { return _table[get_selected_layer_index()]; }
	PVLayer const& get_selected_layer() const { return _table[get_selected_layer_index()]; }

	int get_selected_layer_index() const { return _selected_layer_index; }

	void set_selected_layer_index(int index) { _selected_layer_index = index; }
	//
	void process(PVLayer& output_layer, PVRow row_count);
	//
	PVLayer* append_layer(const PVLayer& layer);
	PVLayer* append_new_layer(PVRow row_count, QString const& name = QString());
	PVLayer*
	append_new_layer_from_selection_and_lines_properties(PVSelection const& selection,
	                                                     PVLinesProperties const& lines_properties);
	bool contains_layer(PVLayer* layer) const;

	void compute_selectable_count();

	void delete_by_index(int index);
	void delete_all_layers();
	void delete_selected_layer();

	PVLayer* duplicate_selected_layer(const QString& name);

	void move_layer_down(int index);
	void move_layer_up(int index);
	void move_selected_layer_to(int new_index);
	void move_selected_layer_down();
	void move_selected_layer_up();
	void hide_layers();

  public:
	void copy_details_to_clipboard();

  public:
	void serialize_write(PVCore::PVSerializeObject& so);
	static Inendi::PVLayerStack serialize_read(PVCore::PVSerializeObject& so);

  private:
	int _selected_layer_index;
	QList<PVLayer> _table;
	bool _should_hide_layers = true;
};
}

#endif /* INENDI_PVLAYERSTACK_H */
