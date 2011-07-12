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

#include <pvcore/general.h>
#include <pvcore/PVListFloat2D.h>

#include <pvrush/PVFormat.h>
#include <pvrush/PVNraw.h>

#include <picviz/PVPtrObjects.h>
#ifdef CUDA
#include <picviz/cuda/PVMapped_create_table_cuda.h>
#endif

#include <boost/shared_ptr.hpp>

namespace Picviz {

/**
 * \class PVMapped
 */
class LibExport PVMapped {
public:
	typedef boost::shared_ptr<PVMapped> p_type;
public:
	PVMapped(PVMapping_p parent);
	~PVMapped();

	PVMapping_p mapping;
	PVRoot_p root;

	PVCore::PVListFloat2D table;
	PVCore::PVListFloat2D trans_table;

	int create_table();
	void to_csv();

	PVRush::PVFormat_p get_format();
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	const PVRush::PVNraw::nraw_trans_table& get_trans_nraw() const;
	void clear_trans_nraw();
	PVSource_p get_source_parent();
	PVRoot_p get_root();

	PVRow get_row_count();
	PVCol get_column_count();

	void run_mandatory_mapping(Picviz::PVRoot_p root, pvrow row, pvcol col, QString value, float mapped_position, bool is_first, void *userdata);
};

typedef PVMapped::p_type PVMapped_p;

}

#endif	/* PICVIZ_PVMAPPED_H */
