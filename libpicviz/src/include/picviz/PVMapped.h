//! \file PVMapped.h
//! $Id: PVMapped.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMAPPED_H
#define PICVIZ_PVMAPPED_H

#include <QList>
#include <QString>
#include <QStringList>
#include <QHash>
#include <QVector>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVListFloat2D.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVPtrObjects.h>
#include <picviz/PVMapping.h>
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
class LibPicvizDecl PVMapped {
	friend class PVPlotted;
	friend class PVSource;
	friend class PVCore::PVSerializeObject;
public:
	typedef boost::shared_ptr<PVMapped> p_type;
	typedef QList<PVPlotted_p> list_plotted_t;
	typedef std::vector< std::pair<PVCol,float> > mapped_sub_col_t;
public:
	PVMapped(PVMapping const& mapping);
	~PVMapped();

	void set_mapping(PVMapping const& mapping);
protected:
	// For serialization
	PVMapped() { }

	// For PVSource
	void invalidate_all();
	
public:
	void process_parent_source();

	void process_from_source(PVSource* src, bool keep_views_info);
	void process_from_parent_source(bool keep_views_info);

	void add_plotted(PVPlotted_p plotted);

	inline bool is_uptodate() const { return _mapping.is_uptodate(); };

	void set_name(QString const& name) { _mapping.set_name(name); }
	QString const& get_name() const { return _mapping.get_name(); }


public:
	// Data access
	PVRow get_row_count();
	PVCol get_column_count();
	void get_sub_col_minmax(mapped_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const;

	list_plotted_t const& get_plotteds() const { return _plotteds; }

public:
	// Debugging functions
	void to_csv();

public:
	// Parents
	PVSource* get_source_parent();
	const PVSource* get_source_parent() const;

	PVRoot* get_root_parent();
	const PVRoot* get_root_parent() const;

	const PVMapping& get_mapping() const { return _mapping; }
	PVMapping& get_mapping() { return _mapping; }

public:
	// NRAW
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	const PVRush::PVNraw::nraw_trans_table& get_trans_nraw() const;
	void clear_trans_nraw();
	
	PVRush::PVFormat_p get_format();

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

private:
	void invalidate_plotted_children_column(PVCol j);
	void create_table();

protected:
	PVMapping _mapping;
	PVRoot* _root;
	PVSource* _source;
	list_plotted_t _plotteds;

	PVCore::PVListFloat2D table;
	PVCore::PVListFloat2D trans_table;
};

typedef PVMapped::p_type PVMapped_p;

}

#endif	/* PICVIZ_PVMAPPED_H */
