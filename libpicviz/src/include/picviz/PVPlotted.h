//! \file PVPlotted.h
//! $Id: PVPlotted.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVPLOTTED_H
#define PICVIZ_PVPLOTTED_H

#include <QList>
#include <QStringList>
#include <QVector>
#include <vector>
#include <utility>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVListFloat2D.h>
#include <pvkernel/rush/PVNraw.h>
#include <picviz/PVPtrObjects.h>
#include <boost/shared_ptr.hpp>


#ifdef CUDA
#include <picviz/cuda/PVPlotted_create_table_cuda.h>
#endif


namespace Picviz {

// Forward declaration
class PVSelection;

/**
 * \class PVPlotted
 */
class LibPicvizDecl PVPlotted {
public:
	typedef boost::shared_ptr<PVPlotted> p_type;
	typedef QVector<float> plotted_table_t;
	typedef std::vector< std::pair<PVCol,float> > plotted_sub_col_t;
public:
	PVPlotted(PVPlotting_p parent);
	~PVPlotted();

	PVPlotting_p plotting;
	PVRoot_p root;

	// TODO: use PVListFloat2D here !
	plotted_table_t table; /* Unidimensionnal. It must be contiguous in memory */
	#ifndef CUDA
	int create_table();
	#else //CUDA
	int create_table_cuda();
	#endif //CUDA
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	PVRow get_row_count() const;
	PVCol get_column_count() const;
	PVSource_p get_source_parent();
	float get_value(PVRow row, PVCol col) const;
	void get_sub_col_minmax(plotted_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const;
	void to_csv();


};

typedef PVPlotted::p_type PVPlotted_p;
}

#endif	/* PICVIZ_PVPLOTTED_H */
