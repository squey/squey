/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef INENDI_PVLAYERSTACK_H
#define INENDI_PVLAYERSTACK_H

#include <pvbase/types.h> // for PVRow

#include <QList>   // for QList
#include <QString> // for QString

namespace Inendi
{
class PVLayer;
} // namespace Inendi
namespace Inendi
{
class PVLinesProperties;
} // namespace Inendi
namespace Inendi
{
class PVSelection;
} // namespace Inendi
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

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
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static Inendi::PVLayerStack serialize_read(PVCore::PVSerializeObject& so);

  private:
	int _selected_layer_index;
	QList<PVLayer> _table;
	bool _should_hide_layers = true;
};
} // namespace Inendi

#endif /* INENDI_PVLAYERSTACK_H */
