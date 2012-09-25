/**
 * \file PVPlotting.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVPLOTTING_H
#define PICVIZ_PVPLOTTING_H

#include <QList>
#include <QStringList>
#include <QLibrary>
#include <QVector>

#include <pvkernel/core/PVDataTreeObject.h>
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
class LibPicvizDecl PVPlotting {
	friend class PVCore::PVSerializeObject;
	friend class PVPlotted;
public:
	typedef boost::shared_ptr<PVPlotting> p_type;
public:
	/**
	 * Constructor
	 */
	PVPlotting(PVPlotted* mapped);
	
	/**
	 * Destructor
	 */
	~PVPlotting();

protected:
	// Serialization
	PVPlotting();
	void serialize(PVCore::PVSerializeObject &so, PVCore::PVSerializeArchive::version_t v);

	// For PVPlotted
	void set_uptodate_for_col(PVCol j);
	void invalidate_column(PVCol j);
	void add_column(PVPlottingProperties const& props);

public:
	// Parents
	
	/**
	 * Gets the associated format
	 */
	PVRush::PVFormat_p get_format() const;

	PVPlotted* get_plotted() { return _plotted; }

	QString const& get_column_type(PVCol col) const;

	bool is_uptodate() const;

	void reset_from_format(PVRush::PVFormat const& format);

public:
	// Data access
	Picviz::PVPlottingFilter::p_type get_filter_for_col(PVCol col);
	PVPlottingProperties const& get_properties_for_col(PVCol col) const { assert(col < _columns.size()); return _columns.at(col); }
	PVPlottingProperties& get_properties_for_col(PVCol col) { assert(col < _columns.size()); return _columns[col]; }
	bool is_col_uptodate(PVCol j) const;
	void set_type_for_col(QString const& type, PVCol col);

	QString const& get_name() const { return _name; }
	void set_name(QString const& name) { _name = name; }

protected:
	QList<PVPlottingProperties> _columns;

	PVPlotted* _plotted;
	QString _name;
};

typedef PVPlotting::p_type PVPlotting_p;

}

#endif	/* PICVIZ_PVPLOTTING_H */
