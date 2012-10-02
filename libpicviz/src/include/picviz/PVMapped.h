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

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDecimalStorage.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVHugePODVector.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVPtrObjects.h>
#include <picviz/PVMapped_types.h>
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
	typedef Picviz::mapped_decimal_storage_type decimal_storage_type;
	typedef std::vector< std::pair<PVRow, decimal_storage_type> > mapped_sub_col_t;
	typedef children_t list_plotted_t;
	//typedef PVCore::PVMatrix<decimal_storage_type, PVCol, PVRow> mapped_table_t;
	typedef PVCore::PVHugePODVector<decimal_storage_type, 16> mapped_row_t;
	typedef std::vector<mapped_row_t> mapped_table_t;

protected:
	PVMapped();

public:
	~PVMapped();

	// For PVSource
	void invalidate_all();
	void validate_all();
	void add_column(PVMappingProperties const& props);
	
public:
	void process_parent_source();
	void process_from_parent_source();

	inline bool is_uptodate() const { return _mapping->is_uptodate(); };

	PVMapping* get_mapping() { return _mapping.get(); }
	void set_mapping(PVMapping* mapping) { _mapping = PVMapping_p(mapping); }
	void set_name(QString const& name) { _mapping->set_name(name); }
	QString const& get_name() const { return _mapping->get_name(); }

	QList<PVCol> get_columns_indexes_values_within_range(decimal_storage_type const min, decimal_storage_type const max, double rate = 1.0);
	QList<PVCol> get_columns_indexes_values_not_within_range(decimal_storage_type const min, decimal_storage_type const max, double rate = 1.0);

	virtual QString get_serialize_description() const { return "Mapping: " + get_name(); }

	inline PVCore::DecimalType get_decimal_type_of_col(PVCol const j) const { return _mapping->get_decimal_type_of_col(j); }
	inline bool is_mapping_pure(PVCol const c) const { return _mapping->is_mapping_pure(c); }

	void init_pure_mapping_functions(PVFilter::PVPureMappingProcessing::list_pure_mapping_t& funcs);

	bool is_current_mapped() const;

protected:
	// This is accessed by PVSource !
	void init_process_from_rush_pipeline();
	void finish_process_from_rush_pipeline();
	void process_rush_pipeline_chunk(PVCore::PVChunk const* chunk, PVRow const cur_r);

public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_column_count() const;
	void get_sub_col_minmax(mapped_sub_col_t& ret, decimal_storage_type& min, decimal_storage_type& max, PVSelection const& sel, PVCol col) const;

	inline decimal_storage_type get_value(PVRow row, PVCol col) const { return _trans_table[col][row]; }

	inline decimal_storage_type* get_column_pointer(PVCol col) { return &_trans_table[col][0]; }
	inline decimal_storage_type const* get_column_pointer(PVCol col) const { return &_trans_table[col][0]; }

	inline mapped_table_t const& get_table() const { return _trans_table; }

public:
	// Debugging functions
	void to_csv();

public:
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
	void allocate_table(PVRow const nrows, PVCol const ncols);
	void reallocate_table(PVRow const nrows);

protected:
	mapped_table_t _trans_table;
	PVMapping_p _mapping;
	std::vector<PVMappingFilter::p_type> _mapping_filters_rush;
	// This is a hash whose key is "group_type", that contains the PVArgument
	// passed through all mapping filters that have the same group and type
	QHash<QString, PVCore::PVArgument> _grp_values_rush;
};

typedef PVMapped::p_type  PVMapped_p;
typedef PVMapped::wp_type PVMapped_wp;

}

#endif	/* PICVIZ_PVMAPPED_H */
