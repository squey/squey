//! \file PVPlotting.h
//! $Id: PVPlotting.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVPLOTTING_H
#define PICVIZ_PVPLOTTING_H

#include <QList>
#include <QStringList>
#include <QLibrary>
#include <QVector>

#include <pvkernel/rush/PVFormat.h>

#include <picviz/general.h>
#include <picviz/PVPlottingProperties.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVPlottingFilter.h>

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace Picviz {

class PVMapped;

#ifndef picviz_plotting_function
#define picviz_plotting_function "picviz_plotting_function"
#endif

/* typedef float (*picviz_plotting_function)(void *plotting, int index, QString &value, void *userdata, int is_first); */

/**
 * \class PVPlotting
 */
class LibPicvizDecl PVPlotting : public boost::enable_shared_from_this<PVPlotting> {
public:
	typedef boost::shared_ptr<PVPlotting> p_type;
public:
	PVMapped_p mapped;
	PVRoot* root;
	QList<PVPlottingProperties> columns;

	/**
	 * Constructor
	 */
	PVPlotting(PVMapped_p parent);
	
	/**
	 * Destructor
	 */
	~PVPlotting();

	/**
	 * Gets the associated format
	 */
	PVRush::PVFormat_p get_format() const;

	/**
	 * Gets the NRAW
	 */
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	
	/**
	 * Gets the parent Picviz::PVSource
	 */
	Picviz::PVSource* get_source_parent();

	Picviz::PVPlottingFilter::p_type get_filter_for_col(PVCol col);
};

typedef PVPlotting::p_type PVPlotting_p;

}

#endif	/* PICVIZ_PVPLOTTING_H */
