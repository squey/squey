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

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <picviz/general.h>
#include <picviz/PVPlottingProperties.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVPlottingFilter.h>

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace Picviz {

class PVMapped;

/**
 * \class PVPlotting
 */
class LibPicvizDecl PVPlotting : public boost::enable_shared_from_this<PVPlotting> {
	friend class PVCore::PVSerializeObject;
	friend class PVPlotted;
public:
	typedef boost::shared_ptr<PVPlotting> p_type;
public:
	/**
	 * Constructor
	 */
	PVPlotting(PVMapped* mapped);
	
	/**
	 * Destructor
	 */
	~PVPlotting();

protected:
	// Serialization
	PVPlotting() { }
	void serialize(PVCore::PVSerializeObject &so, PVCore::PVSerializeArchive::version_t v);

public:
	// Parents
	
	/**
	 * Gets the NRAW
	 */
	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	
	/**
	 * Gets the parent Picviz::PVSource
	 */
	Picviz::PVSource* get_source_parent();
	Picviz::PVRoot* get_root_parent();

	/**
	 * Gets the associated format
	 */
	PVRush::PVFormat_p get_format() const;

	PVMapped* get_mapped_parent();
	const PVMapped* get_mapped_parent() const;

	void set_mapped(PVMapped* mapped);

	QString const& get_column_type(PVCol col) const;


public:
	// Data access
	Picviz::PVPlottingFilter::p_type get_filter_for_col(PVCol col);
	PVPlottingProperties const& get_properties_for_col(PVCol col) { assert(col < _columns.size()); return _columns.at(col); }

protected:
	PVMapped* _mapped;
	PVRoot* _root;
	QList<PVPlottingProperties> _columns;
};

typedef PVPlotting::p_type PVPlotting_p;

}

#endif	/* PICVIZ_PVPLOTTING_H */
