/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVMAPPED_H
#define INENDI_PVMAPPED_H

#include <QString>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDecimalStorage.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVHugePODVector.h>

#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVPtrObjects.h>
#include <inendi/PVMapped_types.h>
#include <inendi/PVMapping.h>
#include <inendi/PVSource.h>

namespace Inendi
{

class PVPlotted;
class PVSelection;

/**
 * \class PVMapped
 *
 * PVMapped is child of PVSource and its children are PVPlotted.
 *
 * It is mainly a proxy class forwarding function to DataTree (for general purpose function)
 * PVMapping for others.
 * It contains only mapping values which certainly should be merged in PVMapping.
 */
typedef typename PVCore::PVDataTreeObject<PVSource, PVPlotted> data_tree_mapped_t;
class PVMapped : public data_tree_mapped_t
{
	friend class PVPlotted;
	friend class PVSource;
	friend class PVCore::PVSerializeObject;

  public:
	using decimal_storage_type = Inendi::mapped_decimal_storage_type;
	using mapped_row_t = PVCore::PVHugePODVector<decimal_storage_type, 16>;
	using mapped_table_t = std::vector<mapped_row_t>;

  public:
	PVMapped();

	/**
	 * Remove its children first.
	 *
	 * @fixme : should be done by default in the datatree.
	 */
	~PVMapped();

	// For PVSource
	void add_column(PVMappingProperties const& props);

  public:
	/**
	 * Compute mapping and chain to plottings.
	 */
	void process_from_parent_source();

	inline bool is_uptodate() const { return _mapping->is_uptodate(); };

	/**
	 * Accessors and modifiers for Mapping.
	 */
	PVMapping* get_mapping() { return _mapping.get(); }
	const PVMapping* get_mapping() const { return _mapping.get(); }
	void set_mapping(PVMapping* mapping) { _mapping = PVMapping_p(mapping); }
	void set_name(QString const& name) { _mapping->set_name(name); }
	QString const& get_name() const { return _mapping->get_name(); }

	/**
	 * Provide decimal type for each column.
	 *
	 * Use to compute plotting.
	 */
	inline PVCore::DecimalType get_decimal_type_of_col(PVCol const j) const
	{
		return _mapping->get_decimal_type_of_col(j);
	}

	/**
	 * Whether it is the current display mapped information.
	 *
	 * @fixme : As we do nothing in this case, it should not be possible to trigger this function
	 *with incorrect mapped.
	 */
	bool is_current_mapped() const;

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
	inline decimal_storage_type get_value(PVRow row, PVCol col) const
	{
		return _trans_table[col][row];
	}
	inline decimal_storage_type const* get_column_pointer(PVCol col) const
	{
		return &_trans_table[col][0];
	}

  private:
	inline decimal_storage_type* get_column_pointer(PVCol col) { return &_trans_table[col][0]; }

  public:
	// Debugging functions
	void to_csv() const;

  protected:
	/**
	 * Set a new parent rebuilding mapping.
	 *
	 * @fixme : It should disappear as it require rebuilding and it means we may have node without
	 *parent.
	 */
	virtual void set_parent_from_ptr(PVSource* source);

	/**
	 * Submode information.
	 */
	virtual QString get_children_description() const { return "Plotted(s)"; }
	virtual QString get_children_serialize_name() const { return "plotted"; }

  public:
	virtual QString get_serialize_description() const { return "Mapping: " + get_name(); }

  protected:
	void serialize_write(PVCore::PVSerializeObject& so);
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	PVSERIALIZEOBJECT_SPLIT

  private:
	/**
	 * Mark plotted as invalid as they will need to be recomputed.
	 */
	void invalidate_plotted_children_column(PVCol j);

	/**
	 * Allocate mapping table for given number fo column and row.
	 */
	void allocate_table(PVRow const nrows, PVCol const ncols);

  protected:
	mapped_table_t _trans_table; //!< This is a vector of vector which contains "for each column"
	// mapping of cell.
	PVMapping_p _mapping; //!< Contains properties for every column.
};

using PVMapped_p = PVMapped::p_type;
using PVMapped_wp = PVMapped::wp_type;
}

#endif /* INENDI_PVMAPPED_H */
