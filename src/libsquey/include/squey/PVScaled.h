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
#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVColumnIndexes.h>
#include <pvkernel/core/PVDataTreeObject.h> // for PVDataTreeChild, etc

#include <pvbase/types.h> // for PVRow, PVCol, etc

#include <QList>
#include <QString>

#include <vector>
#include <utility>
#include <limits>
#include <optional>

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
	/**
	 * Selection scaling: spreading the selected rows over the whole axis.
	 *
	 * A column is normally scaled over the bounds of its every row, so a narrow
	 * selection lands on a sliver of the axis. Under selection scaling a column's
	 * bounds are those of the selection instead, which spreads it over the whole
	 * axis while its own mode -- min/max, logarithmic, uniform -- keeps saying how
	 * far apart the rows sit within it. Rows outside those bounds are pinned to
	 * the ends of the axis rather than dropped.
	 *
	 * The setting below is the scaling's own; a column may override it, see
	 * PVScalingProperties::set_selection_scaling.
	 */
	void set_scale_on_selection(bool enabled) { _scale_on_selection = enabled; }
	bool scale_on_selection() const { return _scale_on_selection; }

	/**
	 * Whether the given column scales over the selection, its own stance settled
	 * against the scaling's setting.
	 */
	bool scale_on_selection(PVCol col) const
	{
		return get_properties_for_col(col).scales_on_selection(_scale_on_selection);
	}

	/**
	 * Whether any column at all scales over the selection.
	 *
	 * False when the setting is off and no column overrides it, in which case a
	 * request to rescale would have nothing to act on.
	 */
	bool scale_on_selection_anywhere() const;

	/**
	 * Whether the scaling follows every change of selection on its own.
	 *
	 * Held here so that it is saved with the investigation and reachable from the
	 * views; acting on it is up to whoever observes the selection.
	 */
	void set_auto_scale_on_selection(bool enabled) { _auto_scale_on_selection = enabled; }
	bool auto_scale_on_selection() const { return _auto_scale_on_selection; }

	/**
	 * Recompute the columns that scale over the selection, over these rows.
	 *
	 * Does nothing when no column asks for it. An empty selection leaves nothing
	 * to spread, so it is refused and the columns keep the bounds they have.
	 *
	 * @return whether any column ended up somewhere else on its axis
	 */
	bool update_scaling_on_selection(PVSelection const& sel);

	/**
	 * Give the columns back the bounds of their every row.
	 */
	void clear_selection_domain();

	/**
	 * The rows the columns are currently scaled over, if any.
	 */
	bool has_selection_domain() const { return _selection_domain.has_value(); }

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
		if (col < 0 || col >= (int)_columns.size()) {
			throw std::out_of_range("PVScalingProperties::get_properties_for_col: Invalid column index");
		}
		return *std::next(_columns.begin(), (size_t)col);
	}
	PVScalingProperties& get_properties_for_col(PVCol col)
    {
        return const_cast<PVScalingProperties&>(static_cast<const PVScaled&>(*this).get_properties_for_col(col));
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

	/**
	 * The rows holding a column's smallest and largest positions, among the selected ones.
	 *
	 * Positions and not values: positions are stored inverted, so the smallest one is
	 * the top of the axis and belongs to the largest value. On a tie, the first row.
	 */
	void get_col_minmax(PVRow& min, PVRow& max, PVSelection const& sel, PVCol col) const;

	/**
	 * The rows holding a column's smallest and largest positions.
	 *
	 * As above, over every row.
	 */
	void get_col_minmax(PVRow& min, PVRow& max, PVCol const col) const;

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

  public:
	sigc::signal<void(QList<PVCol>)> _scaled_updated;

  private:
	/**
	 * Recompute the columns the selection domain has a say over.
	 *
	 * That is those that scale over it now -- the domain itself may have just
	 * changed -- and those that were scaled over the previous one, which are owed
	 * the bounds of their every row back. A column whose bounds turn out to be the
	 * ones it already has is dropped along the way, see create_table: it sits
	 * where it sat, and everything built on top of it still holds.
	 *
	 * @return whether any column ended up somewhere else on its axis
	 */
	bool refresh_selection_scaling();

	/**
	 * The bounds the selection domain gives a column, the invalid rows left out.
	 *
	 * Only meaningful with a domain set.
	 */
	pvcop::db::array compute_selection_minmax(PVCol col) const;

	/**
	 * Place one column's rows on its axis.
	 *
	 * Called on several columns at once, so it touches nothing shared beyond the
	 * entry each column owns.
	 *
	 * @return whether the column ended up somewhere else than it already was
	 */
	bool scale_column(PVCol col);

  private:
	scaleds_t _scaleds;
	QList<PVCol> _last_updated_cols; //!< List of column to update for view on this scaled.
	std::vector<MinMax> _minmax_values;
	std::list<PVScalingProperties> _columns;
	std::string _name;
	std::optional<PVSelection> _selection_domain; //!< rows the columns are scaled over, if any
	//! Per column, whether it was scaled over them. Not a vector<bool>: columns are
	//! scaled side by side, and neighbouring bits of one word cannot be written to
	//! from several threads at once.
	std::vector<char> _scaled_on_domain;
	std::vector<pvcop::db::array> _domain_bounds; //!< per column, the domain bounds last used
	std::vector<bool> _domain_only_invalidation;  //!< per column, invalidated by the domain alone
	bool _scale_on_selection = false;
	bool _auto_scale_on_selection = false;
};
} // namespace Squey

#endif /* SQUEY_PVPLOTTED_H */
