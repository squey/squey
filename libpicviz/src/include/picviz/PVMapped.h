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

/**
 * \class PVMapped
 */
class LibPicvizDecl PVMapped {
	friend class PVPlotted;
public:
	typedef boost::shared_ptr<PVMapped> p_type;
	typedef QList<PVPlotted_p> list_plotted_t;
public:
	PVMapped(PVMapping const& mapping);
	~PVMapped();

public:
	void create_table();
	void add_plotted(PVPlotted_p plotted);

public:
	// Data access
	PVRow get_row_count();
	PVCol get_column_count();

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

public:
	// NRAW
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	const PVRush::PVNraw::nraw_trans_table& get_trans_nraw() const;
	void clear_trans_nraw();
	
	PVRush::PVFormat_p get_format();

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
