/**
 * \file PVMapped.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVMAPPED_H
#define PICVIZ_PVMAPPED_H

#include <QList>
#include <QString>
#include <QStringList>
#include <QHash>
#include <QVector>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVListFloat2D.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVPtrObjects.h>
#include <picviz/PVMapping.h>
#include <picviz/PVSource.h>
#ifdef CUDA
#include <picviz/cuda/PVMapped_create_table_cuda.h>
#endif

#include <boost/shared_ptr.hpp>

namespace Picviz {

class PVPlotted;
class PVSelection;

/**
 * \class PVMapped
 */
typedef typename PVCore::PVDataTreeObject<PVSource, PVPlotted> data_tree_mapped_t;
class LibPicvizDecl PVMapped : public data_tree_mapped_t {
	friend class PVPlotted;
	friend class PVSource;
	friend class PVCore::PVSerializeObject;
	friend class PVCore::PVDataTreeAutoShared<PVMapped>;
public:
	//typedef boost::shared_ptr<PVMapped> p_type;
	typedef std::vector< std::pair<PVCol,float> > mapped_sub_col_t;
	typedef children_t list_plotted_t;

protected:
	PVMapped();

public:
	~PVMapped();

	// For PVSource
	void invalidate_all();
	void add_column(PVMappingProperties const& props);
	
public:
	void process_parent_source();

	void process_from_source(PVSource* src, bool keep_views_info);
	void process_from_parent_source(bool keep_views_info);

	inline bool is_uptodate() const { return _mapping->is_uptodate(); };

	PVMapping* get_mapping() { return _mapping.get(); }
	void set_mapping(PVMapping* mapping) { _mapping = PVMapping_p(mapping); }
	void set_name(QString const& name) { _mapping->set_name(name); }
	QString const& get_name() const { return _mapping->get_name(); }

	QList<PVCol> get_columns_indexes_values_within_range(float min, float max, double rate = 1.0);
	QList<PVCol> get_columns_indexes_values_not_within_range(float min, float max, double rate = 1.0);

public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_column_count() const;
	void get_sub_col_minmax(mapped_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const;

	inline float get_value(PVRow row, PVCol col) const { return trans_table.getValue(col, row); }

	inline float* get_column_pointer(PVCol col) { return trans_table.getRowData(col); }

public:
	// Debugging functions
	void to_csv();

public:
	// NRAW
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	const PVRush::PVNraw::nraw_trans_table& get_trans_nraw() const;
	void clear_trans_nraw();
	
	PVRush::PVFormat_p get_format() const;

protected:
	virtual void set_parent_from_ptr(PVSource* source);
	virtual QString get_children_description() const { return "Plotted(s)"; }
	virtual QString get_children_serialize_name() const { return "plotted"; }


protected:
	void serialize_write(PVCore::PVSerializeObject& so);
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	PVSERIALIZEOBJECT_SPLIT

private:
	void invalidate_plotted_children_column(PVCol j);
	void create_table();

protected:
	PVCore::PVListFloat2D trans_table;
	PVMapping_p _mapping;
};

typedef PVMapped::p_type PVMapped_p;

}

#endif	/* PICVIZ_PVMAPPED_H */
