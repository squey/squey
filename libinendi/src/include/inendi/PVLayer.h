/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYER_H
#define INENDI_PVLAYER_H

#include <inendi/PVLinesProperties.h> // for PVLinesProperties
#include <inendi/PVSelection.h>       // for PVSelection

#include <pvbase/types.h> // for PVRow

#include <QMetaType> // for Q_DECLARE_METATYPE
#include <QString>   // for QString

#include <cstddef> // for size_t
#include <vector>  // for vector

namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

constexpr size_t INENDI_LAYER_NAME_MAXLEN = 1000;

namespace Inendi
{

class PVPlotted;

/**
 * \class PVLayer
 */
class PVLayer
{
  public:
	typedef std::vector<PVRow> list_row_indexes_t;

  private:
	int index;
	bool _locked;
	bool visible;
	QString name;
	PVSelection selection;
	PVLinesProperties lines_properties;

	// Below values are cached values.
	PVRow selectable_count;
	list_row_indexes_t _row_mins;
	list_row_indexes_t _row_maxs;

  public:
	/**
	 * Constructor
	 */
	PVLayer(const QString& name_, size_t size);
	PVLayer(const QString& name_, const PVSelection& sel_, const PVLinesProperties& lp_);

	/**
	 * Copy this layer properties as b properties for selected elements (from selection)
	 * and & selection between current and b layer.
	 *
	 * @param[in] : nelts is the number of elements in the selection.
	 */
	void A2B_copy_restricted_by_selection(PVLayer& b, PVSelection const& selection);

	int get_index() const { return index; }
	const PVLinesProperties& get_lines_properties() const { return lines_properties; }
	PVLinesProperties& get_lines_properties() { return lines_properties; }
	const QString& get_name() const { return name; }
	const PVSelection& get_selection() const { return selection; }
	PVSelection& get_selection() { return selection; }
	bool get_visible() const { return visible; }

	void compute_selectable_count();
	PVRow get_selectable_count() const { return selectable_count; }

	void compute_min_max(PVPlotted const& plotted);
	inline list_row_indexes_t get_mins() const { return _row_mins; }
	inline list_row_indexes_t const& get_maxs() const { return _row_maxs; }

	void reset_to_empty_and_default_color();
	void reset_to_full_and_default_color();
	void reset_to_default_color();

	void set_index(int index_) { index = index_; }

	void set_lock() { _locked = true; }
	bool is_locked() const { return _locked; }

	void set_name(const QString& name_)
	{
		name = name_;
		name.truncate(INENDI_LAYER_NAME_MAXLEN);
	}
	void set_visible(bool visible_) { visible = visible_; }

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static Inendi::PVLayer serialize_read(PVCore::PVSerializeObject& so);
};
} // namespace Inendi

// This must be done outside of any namespace
// This metatype is used for PVLayer widget selection.
Q_DECLARE_METATYPE(Inendi::PVLayer*)

#endif /* INENDI_PVLAYER_H */
