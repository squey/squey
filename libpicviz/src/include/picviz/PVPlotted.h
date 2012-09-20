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
#include <boost/shared_ptr.hpp>
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
		boost::shared_ptr<PVSelection> sel_p;
		QString type;
		void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
		{
			so.attribute("column", col);
			so.attribute("type", type);
			so.object("selection", sel_p, QString(), false, (PVSelection*) NULL, false);
		}
	};
public:
	typedef std::vector<float> plotted_table_t;
	typedef std::vector<uint32_t, PVCore::PVAlignedAllocator<uint32_t, 16> > uint_plotted_table_t;
	typedef std::vector< std::pair<PVCol,uint32_t> > plotted_sub_col_t;
	typedef std::list<ExpandedSelection> list_expanded_selection_t;

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

public:
	// Parents
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;

	PVRush::PVNraw& get_rushnraw_parent();
	const PVRush::PVNraw& get_rushnraw_parent() const;

	uint_plotted_table_t& get_uint_plotted() { return _uint_table; }
	uint_plotted_table_t const& get_uint_plotted() const { return _uint_table; }

	PVPlotting& get_plotting() { return *_plotting; }
	const PVPlotting& get_plotting() const { return *_plotting; }

	bool is_uptodate() const;

	QList<PVCol> get_singleton_columns_indexes();
	QList<PVCol> get_columns_indexes_values_within_range(uint32_t min, uint32_t max, double rate = 1.0);
	QList<PVCol> get_columns_indexes_values_not_within_range(uint32_t min, uint32_t max, double rate = 1.0);

public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_column_count() const;
	inline PVRow get_aligned_row_count() const
	{
		const PVRow ret = get_row_count();
		return ((ret+PVROW_VECTOR_ALIGNEMENT-1)/(PVROW_VECTOR_ALIGNEMENT))*PVROW_VECTOR_ALIGNEMENT;
	}
	PVSource* get_source_parent();
	inline uint32_t const* get_column_pointer(PVCol const j) const { return &_uint_table[j*get_aligned_row_count()]; }
	inline uint32_t* get_column_pointer(PVCol const j) { return &_uint_table[j*get_aligned_row_count()]; }
	inline uint32_t get_value(PVRow const i, PVCol const j) const { return get_column_pointer(j)[i]; }

	void get_sub_col_minmax(plotted_sub_col_t& ret, uint32_t& min, uint32_t& max, PVSelection const& sel, PVCol col) const;
	void get_col_minmax(PVRow& min, PVRow& max, PVSelection const& sel, PVCol col) const;

	inline PVView* current_view() { return get_parent<PVSource>()->current_view(); }
	inline const PVView* current_view() const { return get_parent<PVSource>()->current_view(); }
	void expand_selection_on_axis(PVSelection const& sel, PVCol axis_id, QString const& mode, bool add = true);

	// Plotted dump/load
	bool dump_buffer_to_file(QString const& file, bool write_as_transposed = false) const;
	static bool load_buffer_from_file(uint_plotted_table_t& buf, PVRow& nrows, PVCol& ncols, bool get_transposed_version, QString const& file);
	static bool load_buffer_from_file(plotted_table_t& buf, PVCol& ncols, bool get_transposed_version, QString const& file);

public:
	// Debug
	void to_csv();

protected:
	virtual void set_parent_from_ptr(PVMapped* mapped);
	virtual QString get_children_description() const { return "View(s)"; }
	virtual QString get_children_serialize_name() const { return "views"; }
	virtual void child_added(PVView& child);

	int create_table();

private:
	PVPlotting_p _plotting;
	uint_plotted_table_t _uint_table;
	list_expanded_selection_t _expanded_sels;
};

typedef PVPlotted::p_type  PVPlotted_p;
typedef PVCore::PVSharedPtr<PVPlotted> PVPlotted_sp;

}

#endif	/* PICVIZ_PVPLOTTED_H */
