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
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVNraw.h>
#include <picviz/PVPtrObjects.h>
#include <boost/shared_ptr.hpp>
#include <picviz/PVView_types.h>
#include <picviz/PVPlotting.h>


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
	friend class PVCore::PVSerializeObject;
public:
	typedef boost::shared_ptr<PVPlotted> p_type;
	typedef QVector<float> plotted_table_t;
	typedef std::vector< std::pair<PVCol,float> > plotted_sub_col_t;
public:
	PVPlotted(PVPlotting const& plotting);
	~PVPlotted();

protected:
	// Serialization
	PVPlotted() { }
	void serialize(PVCore::PVSerializeObject &so, PVCore::PVSerializeArchive::version_t v);

public:
	#ifndef CUDA
	int create_table();
	#else //CUDA
	int create_table_cuda();
	#endif //CUDA

	void process_from_mapped(PVMapped* mapped, bool keep_views_info);

public:
	// Parents
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;

	PVRush::PVNraw& get_rushnraw_parent();
	const PVRush::PVNraw& get_rushnraw_parent() const;

	PVRoot* get_root_parent() { return root; }
	const PVRoot* get_root_parent() const { return root; }

	PVMapped* get_mapped_parent() { return _mapped; }
	const PVMapped* get_mapped_parent() const { return _mapped; }

	const float* get_table_pointer() const { return &_table.at(0); }

	PVPlotting& get_plotting() { return _plotting; }
	const PVPlotting& get_plotting() const { return _plotting; }

public:
	// Data access
	PVRow get_row_count() const;
	PVCol get_column_count() const;
	PVSource* get_source_parent();
	float get_value(PVRow row, PVCol col) const;
	void get_sub_col_minmax(plotted_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const;
	inline plotted_table_t const& get_table() const { return _table; }
	inline PVView_p get_view() { return _view; }
	inline const PVView_p get_view() const { return _view; }

public:
	// Debug
	void to_csv();

private:
	void set_plotting(PVPlotting const& plotting);

private:
	PVPlotting _plotting;
	PVRoot* root;
	PVMapped* _mapped;
	plotted_table_t _table; /* Unidimensionnal. It must be contiguous in memory */
	PVView_p _view;
};

typedef PVPlotted::p_type PVPlotted_p;
}

#endif	/* PICVIZ_PVPLOTTED_H */
