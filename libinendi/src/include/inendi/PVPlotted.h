/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVPLOTTED_H
#define INENDI_PVPLOTTED_H

#include <QList>
#include <QStringList>
#include <QVector>
#include <vector>
#include <utility>
#include <limits>

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVHugePODVector.h>
#include <pvkernel/rush/PVNraw.h>
#include <inendi/PVView.h>
#include <inendi/PVPlotting.h>
#include <inendi/PVSelection.h>

namespace Inendi
{

// Forward declaration
class PVMapped;
class PVSource;

/**
 * \class PVPlotted
 */
class PVPlotted : public PVCore::PVDataTreeChild<PVMapped, PVPlotted>,
                  public PVCore::PVDataTreeParent<PVView, PVPlotted>
{
	friend class PVCore::PVSerializeObject;
	friend class PVMapped;
	friend class PVSource;

  public:
	using value_type = uint32_t;
	static constexpr value_type MAX_VALUE = std::numeric_limits<value_type>::max();

  private:
	struct MinMax {
		PVRow min;
		PVRow max;
	};

  public:
	typedef std::vector<float> plotted_table_t;
	typedef PVCore::PVHugePODVector<uint32_t, 16> uint_plotted_table_t;
	typedef std::vector<std::pair<PVCol, uint32_t>> plotted_sub_col_t;
	typedef std::vector<PVRow> rows_vector_t;

  public:
	PVPlotted(PVMapped& mapped);

  public:
	~PVPlotted();

  protected:
	// Serialization
	void serialize_write(PVCore::PVSerializeObject& so);
	void serialize_read(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

	// For PVMapped
	inline void invalidate_column(PVCol j) { return _plotting.invalidate_column(j); }

  public:
	void update_plotting();

	void set_name(std::string const& name) { _plotting.set_name(name); }
	std::string const& get_name() const { return _plotting.get_name(); }

	std::string get_serialize_description() const override { return "Plotting: " + get_name(); }

	bool is_current_plotted() const;

	/**
	 * do any process after a mapped load
	 */
	void finish_process_from_rush_pipeline();

  public:
	PVRush::PVNraw& get_rushnraw_parent();
	const PVRush::PVNraw& get_rushnraw_parent() const;

	uint_plotted_table_t& get_uint_plotted() { return _uint_table; }
	uint_plotted_table_t const& get_uint_plotted() const { return _uint_table; }

	PVPlotting& get_plotting() { return _plotting; }
	const PVPlotting& get_plotting() const { return _plotting; }

	inline PVPlottingProperties const& get_plotting_properties(PVCol j)
	{
		assert(j < get_column_count());
		return _plotting.get_properties_for_col(j);
	}

	bool is_uptodate() const;

	QList<PVCol> get_singleton_columns_indexes();
	QList<PVCol>
	get_columns_indexes_values_within_range(uint32_t min, uint32_t max, double rate = 1.0);
	QList<PVCol>
	get_columns_indexes_values_not_within_range(uint32_t min, uint32_t max, double rate = 1.0);
	QList<PVCol> get_columns_to_update() const;

  public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_column_count() const;

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

	PVSource* get_source_parent();
	inline uint32_t const* get_column_pointer(PVCol const j) const
	{
		return &_uint_table[get_plotted_col_offset(get_row_count(), j)];
	}
	inline uint32_t* get_column_pointer(PVCol const j)
	{
		return &_uint_table[get_plotted_col_offset(get_row_count(), j)];
	}
	inline uint32_t get_value(PVRow const i, PVCol const j) const
	{
		return get_column_pointer(j)[i];
	}

	/**
	 * Returns the base address of a column in a plotted's buffer
	 *
	 * @param plotted the plotted's base address
	 * @param nrows the plotted's rows number
	 * @param col the wanted column number
	 *
	 * @return the base address of the column
	 */
	static const uint32_t*
	get_plotted_col_addr(const uint32_t* plotted, const PVRow nrows, const PVCol col)
	{
		return plotted + get_plotted_col_offset(nrows, col);
	}

	/**
	 * Returns the base address of a column in a plotted
	 *
	 * @param plotted the plotted's base address
	 * @param nrows the plotted's rows number
	 * @param col the wanted column number
	 *
	 * @return the base address of the column
	 */
	static const uint32_t*
	get_plotted_col_addr(const uint_plotted_table_t& plotted, const PVRow nrows, const PVCol col)
	{
		return get_plotted_col_addr(&plotted.at(0), nrows, col);
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

  protected:
	virtual QString get_children_description() const { return "View(s)"; }
	virtual QString get_children_serialize_name() const { return "views"; }

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
	PVPlotting _plotting;
	uint_plotted_table_t _uint_table;
	QList<PVCol> _last_updated_cols;
	std::vector<MinMax> _minmax_values;
};
}

#endif /* INENDI_PVPLOTTED_H */
