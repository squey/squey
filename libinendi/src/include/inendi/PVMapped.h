/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVMAPPED_H
#define INENDI_PVMAPPED_H

#include <QString>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVHugePODVector.h>

#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVMapping.h>
#include <inendi/PVMappingProperties.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>

namespace Inendi
{

class PVPlotted;

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
	PVMapped(PVSource& src);

  public:
	/**
	 * Compute mapping and chain to plottings.
	 */
	void update_mapping();

	inline bool is_uptodate() const { return _mapping.is_uptodate(); };

	/**
	 * Accessors and modifiers for Mapping.
	 */
	PVMapping& get_mapping() { return _mapping; }
	const PVMapping& get_mapping() const { return _mapping; }
	std::string const& get_name() const { return _mapping.get_name(); }

	/**
	 * Ask to compute mapping based on Mapping filter for each column.
	 *
	 * Only "not up to date" mapping will be computer.
	 */
	void compute();

  public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_column_count() const;

	/**
	 * Access mapping value for given row/col.
	 */
	inline pvcop::db::array const& get_column(PVCol col) const { return _trans_table[col]; }

  public:
	/**
	 * Submode information.
	 */
	virtual QString get_children_description() const { return "Plotted(s)"; }
	virtual QString get_children_serialize_name() const { return "plotted"; }

  public:
	virtual std::string get_serialize_description() const { return "Mapping: " + get_name(); }

  public:
	void serialize_write(PVCore::PVSerializeObject& so);
	static Inendi::PVMapped& serialize_read(PVCore::PVSerializeObject& so,
	                                        Inendi::PVSource& parent);

  private:
	/**
	 * Mark plotted as invalid as they will need to be recomputed.
	 */
	void invalidate_plotted_children_column(PVCol j);

  protected:
	mapped_table_t _trans_table; //!< This is a vector of vector which contains "for each column"
	// mapping of cell.
	PVMapping _mapping; //!< Contains properties for every column.
};
}

#endif /* INENDI_PVMAPPED_H */
