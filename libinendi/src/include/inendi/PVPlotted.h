/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVPLOTTED_H
#define INENDI_PVPLOTTED_H

#include <inendi/PVPlottingProperties.h>
#include <inendi/PVView.h>

#include <pvkernel/core/PVColumnIndexes.h>
#include <pvkernel/core/PVDataTreeObject.h> // for PVDataTreeChild, etc

#include <pvbase/types.h> // for PVRow, PVCol, etc

#include <QList>
#include <QString>

#include <vector>
#include <utility>
#include <limits>

#include <sigc++/sigc++.h>

#include <cassert>  // for assert
#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t
#include <iterator> // for advance
#include <list>     // for _List_const_iterator, list, etc
#include <string>   // for string, operator+

namespace Inendi
{
class PVMapped;
} // namespace Inendi
namespace Inendi
{
class PVSelection;
} // namespace Inendi
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore
namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace Inendi
{

/**
 * \class PVPlotted
 */
class PVPlotted : public PVCore::PVDataTreeChild<PVMapped, PVPlotted>,
                  public PVCore::PVDataTreeParent<PVView, PVPlotted>
{
  public:
	using value_type = uint32_t;
	using plotted_t = pvcop::db::array;
	using uint_plotted_t = pvcop::core::array<value_type>;
	using plotteds_t = std::vector<plotted_t>;

	static constexpr value_type MAX_VALUE = std::numeric_limits<value_type>::max();

  private:
	struct MinMax {
		PVRow min;
		PVRow max;
	};

  public:
	explicit PVPlotted(PVMapped& mapped, std::string const& name = "default");
	PVPlotted(PVMapped& mapped,
	          std::list<Inendi::PVPlottingProperties>&& column,
	          std::string const& name = "default");

  public:
	~PVPlotted();

  public:
	// Serialization
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static Inendi::PVPlotted& serialize_read(PVCore::PVSerializeObject& so,
	                                         Inendi::PVMapped& parent);

	// For PVMapped
	inline void invalidate_column(PVCol j) { return get_properties_for_col(j).invalidate(); }

  public:
	void update_plotting();
	bool is_uptodate() const;

	void set_name(std::string const& name) { _name = name; }
	std::string const& get_name() const { return _name; }

	std::string get_serialize_description() const override { return "Plotting: " + get_name(); }

  public:
	PVRush::PVNraw& get_rushnraw_parent();
	const PVRush::PVNraw& get_rushnraw_parent() const;

	plotteds_t const& get_plotteds() const { return _plotteds; }
	uint_plotted_t const& get_plotted(PVCol col) const
	{
		return _plotteds[col].to_core_array<value_type>();
	}

	PVPlottingProperties const& get_properties_for_col(PVCol col) const
	{
		assert((size_t)col < _columns.size());
		auto begin = _columns.begin();
		std::advance(begin, col);
		return *begin;
	}
	PVPlottingProperties& get_properties_for_col(PVCol col)
	{
		assert((size_t)col < _columns.size());
		auto begin = _columns.begin();
		std::advance(begin, col);
		return *begin;
	}

	QList<PVCol> get_singleton_columns_indexes();
	QList<PVCol>
	get_columns_indexes_values_within_range(uint32_t min, uint32_t max, double rate = 1.0);
	QList<PVCol>
	get_columns_indexes_values_not_within_range(uint32_t min, uint32_t max, double rate = 1.0);
	QList<PVCol> get_columns_to_update() const;

  public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_nraw_column_count() const;

	/**
	 * Returns the aligned row count given a row count
	 *
	 * @param nrows the rows number
	 *
	 * @return the aligned row count corresponding to nrows
	 */
	static PVRow get_aligned_row_count(const PVRow nrows)
	{
		return ((nrows + PVROW_VECTOR_ALIGNEMENT - 1) / (PVROW_VECTOR_ALIGNEMENT)) *
		       PVROW_VECTOR_ALIGNEMENT;
	}

	/**
	 * Returns the aligned row count of this plotted
	 *
	 * @return the corresponding aligned row count
	 */
	inline PVRow get_aligned_row_count() const { return get_aligned_row_count(get_row_count()); }

	inline uint32_t const* get_column_pointer(PVCol const j) const
	{
		return &_plotteds[j].to_core_array<value_type>()[0];
	}
	inline uint32_t get_value(PVRow const i, PVCol const j) const
	{
		return get_column_pointer(j)[i];
	}

	void get_col_minmax(PVRow& min, PVRow& max, PVSelection const& sel, PVCol col) const;

	/** get_col_minmax
	 *
	 * Compute row indices for containing min and max value for a given column.
	 *
	 * @param[out] min : Minimum value of the column
	 * @param[out] max : Maximum value of the column
	 * @param[in] col : Column where we want to extra minmax
	 */
	void get_col_minmax(PVRow& min, PVRow& max, PVCol const col) const;

	inline QList<PVCol> const& last_updated_cols() const { return _last_updated_cols; }

	PVRow get_col_min_row(PVCol const c) const;
	PVRow get_col_max_row(PVCol const c) const;

	std::string export_line(PVRow idx,
	                        const PVCore::PVColumnIndexes& col_indexes,
	                        const std::string sep_char,
	                        const std::string) const;

  private:
	inline uint32_t* get_column_pointer(PVCol const j)
	{
		return &_plotteds[j].to_core_array<value_type>()[0];
	}

  protected:
	int create_table();

	/**
	 * Computes the offset between the plotting's base address and a column's base address
	 *
	 * @param nrows the rows number
	 * @param colo the wanted column index
	 *
	 * @return the offset of the @a col in the plotted's buffer
	 */
	static inline size_t get_plotted_col_offset(PVRow nrows, PVCol col)
	{
		return (size_t)get_aligned_row_count(nrows) * (size_t)col;
	}

  public:
	sigc::signal<void> _plotted_updated;

  private:
	plotteds_t _plotteds;
	QList<PVCol> _last_updated_cols; //!< List of column to update for view on this plotted.
	std::vector<MinMax> _minmax_values;
	std::list<PVPlottingProperties> _columns;
	std::string _name;
};
} // namespace Inendi

#endif /* INENDI_PVPLOTTED_H */
