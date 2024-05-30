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

#ifndef SQUEY_PVPLOTTED_H
#define SQUEY_PVPLOTTED_H

#include <squey/PVScalingProperties.h>
#include <squey/PVView.h>

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

namespace Squey
{
class PVMapped;
} // namespace Squey
namespace Squey
{
class PVSelection;
} // namespace Squey
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore
namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace Squey
{

/**
 * \class PVScaled
 */
class PVScaled : public PVCore::PVDataTreeChild<PVMapped, PVScaled>,
                  public PVCore::PVDataTreeParent<PVView, PVScaled>
{
  public:
	using value_type = uint32_t;
	using scaled_t = pvcop::db::array;
	using uint_scaled_t = pvcop::core::array<value_type>;
	using scaleds_t = std::vector<scaled_t>;

	static constexpr value_type MAX_VALUE = std::numeric_limits<value_type>::max();

  private:
	struct MinMax {
		PVRow min;
		PVRow max;
	};

  public:
	explicit PVScaled(PVMapped& mapped, std::string const& name = "default");
	PVScaled(PVMapped& mapped,
	          std::list<Squey::PVScalingProperties>&& column,
	          std::string const& name = "default");

  public:
	virtual ~PVScaled();

  public:
	// Serialization
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static Squey::PVScaled& serialize_read(PVCore::PVSerializeObject& so,
	                                         Squey::PVMapped& parent);

	// For PVMapped
	inline void invalidate_column(PVCol j) { return get_properties_for_col(j).invalidate(); }

  public:
	void update_scaling();
	bool is_uptodate() const;

	void set_name(std::string const& name) { _name = name; }
	std::string const& get_name() const { return _name; }

  public:
	PVRush::PVNraw& get_rushnraw_parent();
	const PVRush::PVNraw& get_rushnraw_parent() const;

	scaleds_t const& get_scaleds() const { return _scaleds; }
	uint_scaled_t const& get_scaled(PVCol col) const
	{
		return _scaleds[col].to_core_array<value_type>();
	}

	PVScalingProperties const& get_properties_for_col(PVCol col) const
	{
		assert((size_t)col < _columns.size());
		auto begin = _columns.begin();
		std::advance(begin, col);
		return *begin;
	}
	PVScalingProperties& get_properties_for_col(PVCol col)
	{
		assert((size_t)col < _columns.size());
		auto begin = _columns.begin();
		std::advance(begin, col);
		return *begin;
	}

	void append_scaled();
	void delete_scaled(PVCol col);

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
	 * Returns the aligned row count of this scaled
	 *
	 * @return the corresponding aligned row count
	 */
	inline PVRow get_aligned_row_count() const { return get_aligned_row_count(get_row_count()); }

	inline uint32_t const* get_column_pointer(PVCol const j) const
	{
		return &_scaleds[j].to_core_array<value_type>()[0];
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
		return &_scaleds[j].to_core_array<value_type>()[0];
	}

  protected:
	int create_table();

	/**
	 * Computes the offset between the scaling's base address and a column's base address
	 *
	 * @param nrows the rows number
	 * @param colo the wanted column index
	 *
	 * @return the offset of the @a col in the scaled's buffer
	 */
	static inline size_t get_scaled_col_offset(PVRow nrows, PVCol col)
	{
		return (size_t)get_aligned_row_count(nrows) * (size_t)col;
	}

  public:
	sigc::signal<void(QList<PVCol>)> _scaled_updated;

  private:
	scaleds_t _scaleds;
	QList<PVCol> _last_updated_cols; //!< List of column to update for view on this scaled.
	std::vector<MinMax> _minmax_values;
	std::list<PVScalingProperties> _columns;
	std::string _name;
};
} // namespace Squey

#endif /* SQUEY_PVPLOTTED_H */
