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
#include <pvkernel/core/PVListFloat2D.h>
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
	//typedef boost::shared_ptr<PVPlotted> p_type;
	typedef std::vector<float> plotted_table_t;
	typedef std::vector< std::pair<PVCol,float> > plotted_sub_col_t;
	typedef std::list<ExpandedSelection> list_expanded_selection_t;

public:
	PVPlotted();

public:
	~PVPlotted();

	void set_plotting(PVPlotting* plotting) { _plotting = PVPlotting_p(plotting); }

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
	#ifndef CUDA
	int create_table();
	#else //CUDA
	int create_table_cuda();
	#endif //CUDA
	void process_expanded_selections();

	void process_from_mapped(PVMapped* mapped, bool keep_views_info);
	void process_from_parent_mapped(bool keep_views_info);

	void set_name(QString const& name) { _plotting->set_name(name); }
	QString const& get_name() const { return _plotting->get_name(); }
	PVPlotting* get_plotting() { return _plotting.get(); }

public:
	// Parents
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;

	PVRush::PVNraw& get_rushnraw_parent();
	const PVRush::PVNraw& get_rushnraw_parent() const;

	const float* get_table_pointer() const { return &_table.at(0); }

	bool is_uptodate() const;

	QList<PVCol> get_singleton_columns_indexes();
	QList<PVCol> get_columns_indexes_values_within_range(float min, float max, double rate = 1.0);
	QList<PVCol> get_columns_indexes_values_not_within_range(float min, float max, double rate = 1.0);

public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_column_count() const;
	float get_value(PVRow row, PVCol col) const;
	void get_sub_col_minmax(plotted_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const;
	void get_col_minmax(PVRow& min, PVRow& max, PVSelection const& sel, PVCol col) const;
	inline plotted_table_t const& get_table() const { return _table; }
	inline PVView_p get_view() { return _view; }
	inline const PVView_p get_view() const { return _view; }
	void expand_selection_on_axis(PVSelection const& sel, PVCol axis_id, QString const& mode, bool add = true);

	// Plotted dump/load
	bool dump_buffer_to_file(QString const& file, bool write_as_transposed = false) const;
	static bool load_buffer_from_file(plotted_table_t& buf, PVCol& ncols, bool get_transpsed_version, QString const& file);

public:
	// Debug
	void to_csv();

protected:
	virtual void set_parent_from_ptr(PVMapped* mapped);
	virtual QString get_children_description() const { return "View(s)"; }
	virtual QString get_children_serialize_name() const { return "views"; }

private:
	PVPlotting_p _plotting;
	plotted_table_t _table; /* Unidimensionnal. It must be contiguous in memory */
	std::vector<float> _tmp_values;
	PVView_p _view;
	list_expanded_selection_t _expanded_sels;
};

typedef PVPlotted::p_type PVPlotted_p;
}

#endif	/* PICVIZ_PVPLOTTED_H */
