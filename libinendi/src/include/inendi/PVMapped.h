/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVMAPPED_H
#define INENDI_PVMAPPED_H

#include <inendi/PVMappingProperties.h> // for PVMappingProperties
#include <inendi/PVPlotted.h>           // for PVPlotted

#include <pvkernel/core/PVDataTreeObject.h> // for PVDataTreeChild, etc

#include <pvbase/types.h> // for PVCol, PVRow

#include <pvcop/db/array.h> // for array

#include <QString> // for QString

#include <algorithm>  // for all_of
#include <cassert>    // for assert
#include <cstddef>    // for size_t
#include <functional> // for _Mem_fn, mem_fn
#include <iterator>   // for advance
#include <list>       // for _List_const_iterator, list, etc
#include <string>     // for string, operator+
#include <vector>     // for vector

namespace Inendi
{
class PVSource;
} // namespace Inendi
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

namespace Inendi
{

/**
 * \class PVMapped
 *
 * PVMapped is child of PVSource and its children are PVPlotted.
 *
 * It is mainly a proxy class forwarding function to DataTree (for general purpose function)
 * PVMapping for others.
 * It contains only mapping values which certainly should be merged in PVMapping.
 */
class PVMapped : public PVCore::PVDataTreeParent<PVPlotted, PVMapped>,
                 public PVCore::PVDataTreeChild<PVSource, PVMapped>
{
  public:
	using mapped_table_t = std::vector<pvcop::db::array>;

  public:
	explicit PVMapped(PVSource& src, std::string const& name = "default");
	PVMapped(PVSource& src,
	         std::string const& name,
	         std::list<Inendi::PVMappingProperties>&& columns);

  public:
	/**
	 * Compute mapping and chain to plottings.
	 */
	void update_mapping();

	inline bool is_uptodate() const
	{
		return std::all_of(columns.begin(), columns.end(),
		                   std::mem_fn(&PVMappingProperties::is_uptodate));
	}

	/**
	 * Accessors and modifiers for Mapping.
	 */
	std::string const& get_name() const { return _name; }
	void set_name(std::string const& name) { _name = name; }

	PVMappingProperties const& get_properties_for_col(PVCol col) const
	{
		assert((size_t)col < columns.size());
		auto it = columns.begin();
		std::advance(it, col);
		return *it;
	}
	PVMappingProperties& get_properties_for_col(PVCol col)
	{
		assert((size_t)col < columns.size());
		auto it = columns.begin();
		std::advance(it, col);
		return *it;
	}

	void append_mapped();
	void delete_mapped(PVCol col);

	/**
	 * Ask to compute mapping based on Mapping filter for each column.
	 *
	 * Only "not up to date" mapping will be computer.
	 */
	void compute();

  public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_nraw_column_count() const;

	/**
	 * Access mapping value for given row/col.
	 */
	pvcop::db::array const& get_column(PVCol col) const;

  public:
	std::string get_serialize_description() const override { return "Mapping: " + get_name(); }

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static Inendi::PVMapped& serialize_read(PVCore::PVSerializeObject& so,
	                                        Inendi::PVSource& parent);

  private:
	/**
	 * Mark plotted as invalid as they will need to be recomputed.
	 */
	void invalidate_plotted_children_column(PVCol j);

  protected:
	mapped_table_t _trans_table; //!< This is a vector of vector which contains "for each column"
	std::list<PVMappingProperties> columns;
	std::string _name;
};
} // namespace Inendi

#endif /* INENDI_PVMAPPED_H */
