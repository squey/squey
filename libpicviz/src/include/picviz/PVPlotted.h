/**
 * \file PVPlotted.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVPLOTTED_H
#define PICVIZ_PVPLOTTED_H

#include <QList>
#include <QStringList>
#include <QVector>
#include <vector>
#include <utility>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVDecimalStorage.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVNraw.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVMapped.h>
#include <picviz/PVView_types.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVSelection.h>

#ifdef CUDA
#include <picviz/cuda/PVPlotted_create_table_cuda.h>
#endif

namespace Picviz {

// Forward declaration
class PVMapped;
class PVSource;

/**
 * \class PVPlotted
 */
typedef typename PVCore::PVDataTreeObject<PVMapped, PVView> data_tree_plotted_t ;
class LibPicvizDecl PVPlotted : public data_tree_plotted_t {
	friend class PVCore::PVSerializeObject;
	friend class PVMapped;
	friend class PVSource;
private:
	struct ExpandedSelection
	{
		ExpandedSelection(PVCol col_, PVSelection const& sel_, QString const& type_):
			col(col_),
			sel_p(new PVSelection(sel_)),
			type(type_)
		{ }
		ExpandedSelection():
			col(0)
		{ }
		PVCol col;
		std::shared_ptr<PVSelection> sel_p;
		QString type;
		void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
		{
			so.attribute("column", col);
			so.attribute("type", type);
			so.object("selection", sel_p, QString(), false, (PVSelection*) NULL, false);
		}
	};

	struct MinMax
	{
		PVRow min;
		PVRow max;
	};

public:
	typedef std::vector<float> plotted_table_t;
	//typedef std::vector<uint32_t, PVCore::PVAlignedAllocator<uint32_t, 16> > uint_plotted_table_t;
	//typedef std::vector<uint32_t, PVCore::PVNUMAHugePagedInterleavedAllocator<uint32_t> > uint_plotted_table_t;
	typedef PVCore::PVHugePODVector<uint32_t, 16> uint_plotted_table_t;
	typedef std::vector< std::pair<PVCol,uint32_t> > plotted_sub_col_t;
	typedef std::list<ExpandedSelection> list_expanded_selection_t;
	typedef std::vector<PVRow> rows_vector_t;

public:
	PVPlotted();

public:
	~PVPlotted();

protected:
	// Serialization
	void serialize_write(PVCore::PVSerializeObject& so);
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/);
	PVSERIALIZEOBJECT_SPLIT

	// For PVMapped
	inline void invalidate_column(PVCol j) { return _plotting->invalidate_column(j); }

	// For PVSource
	void add_column(PVPlottingProperties const& props);

public:
	void process_expanded_selections();

	void process_parent_mapped();
	void process_from_parent_mapped();

	void set_name(QString const& name) { _plotting->set_name(name); }
	QString const& get_name() const { return _plotting->get_name(); }

	static void norm_int_plotted(plotted_table_t const& trans_plotted, uint_plotted_table_t& res, PVCol ncols);

	void set_plotting(PVPlotting_p const& plotting) { _plotting = plotting; }

	virtual QString get_serialize_description() const { return "Plotting: " + get_name(); }

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

	PVPlotting& get_plotting() { return *_plotting; }
	const PVPlotting& get_plotting() const { return *_plotting; }

	inline PVPlottingProperties const& get_plotting_properties(PVCol j) { assert(j < get_column_count()); return _plotting->get_properties_for_col(j); }

	bool is_uptodate() const;

	QList<PVCol> get_singleton_columns_indexes();
	QList<PVCol> get_columns_indexes_values_within_range(uint32_t min, uint32_t max, double rate = 1.0);
	QList<PVCol> get_columns_indexes_values_not_within_range(uint32_t min, uint32_t max, double rate = 1.0);
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
		return ((nrows+PVROW_VECTOR_ALIGNEMENT-1)/(PVROW_VECTOR_ALIGNEMENT))*PVROW_VECTOR_ALIGNEMENT;
	}

	/**
	 * Returns the aligned row count of this plotted
	 *
	 * @return the corresponding aligned row count
	 */
	inline PVRow get_aligned_row_count() const
	{
		return get_aligned_row_count(get_row_count());
	}

	PVSource* get_source_parent();
	inline uint32_t const* get_column_pointer(PVCol const j) const { return &_uint_table[get_plotted_col_offset(get_row_count(), j)]; }
	inline uint32_t* get_column_pointer(PVCol const j) { return &_uint_table[get_plotted_col_offset(get_row_count(), j)]; }
	inline uint32_t get_value(PVRow const i, PVCol const j) const { return get_column_pointer(j)[i]; }

	/**
	 * Returns the base address of a column in a plotted's buffer
	 *
	 * @param plotted the plotted's base address
	 * @param nrows the plotted's rows number
	 * @param col the wanted column number
	 *
	 * @return the base address of the column
	 */
	static const uint32_t *get_plotted_col_addr(const uint32_t *plotted, const PVRow nrows, const PVCol col)
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
	static const uint32_t *get_plotted_col_addr(const uint_plotted_table_t &plotted, const PVRow nrows, const PVCol col)
	{
		return get_plotted_col_addr(&plotted.at(0), nrows, col);
	}

	void get_sub_col_minmax(plotted_sub_col_t& ret, uint32_t& min, uint32_t& max, PVSelection const& sel, PVCol col) const;
	void get_col_minmax(PVRow& min, PVRow& max, PVSelection const& sel, PVCol col) const;
	void get_col_minmax(PVRow& min, PVRow& max, PVCol const col) const;

	inline PVView* current_view() { return get_parent<PVSource>()->current_view(); }
	inline const PVView* current_view() const { return get_parent<PVSource>()->current_view(); }
	void expand_selection_on_axis(PVSelection const& sel, PVCol axis_id, QString const& mode, bool add = true);

	// Plotted dump/load
	bool dump_buffer_to_file(QString const& file, bool write_as_transposed = false) const;
	static bool load_buffer_from_file(uint_plotted_table_t& buf, PVRow& nrows, PVCol& ncols, bool get_transposed_version, QString const& file);
	static bool load_buffer_from_file(plotted_table_t& buf, PVCol& ncols, bool get_transposed_version, QString const& file);

	inline QList<PVCol> const& last_updated_cols() const { return _last_updated_cols; }

	PVRow get_col_min_row(PVCol const c) const;
	PVRow get_col_max_row(PVCol const c) const;

public:
	// Debug
	void to_csv();

protected:
	virtual void set_parent_from_ptr(PVMapped* mapped);
	virtual QString get_children_description() const { return "View(s)"; }
	virtual QString get_children_serialize_name() const { return "views"; }
	virtual void child_added(PVView& child) override;

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

private:
	PVPlotting_p _plotting;
	uint_plotted_table_t _uint_table;
	list_expanded_selection_t _expanded_sels;
	QList<PVCol> _last_updated_cols;
	std::vector<MinMax> _minmax_values;
};

typedef PVPlotted::p_type  PVPlotted_p;
typedef PVCore::PVSharedPtr<PVPlotted> PVPlotted_sp;

}

#endif	/* PICVIZ_PVPLOTTED_H */
