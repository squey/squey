//! \file PVPlottingProperties.h
//! $Id: PVPlottingProperties.h 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVPLOTTINGPROPERTIES_H
#define PICVIZ_PVPLOTTINGPROPERTIES_H

//#include <QList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVRoot.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

/**
* \class PVPlottingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class LibPicvizDecl PVPlottingProperties {
	friend class PVCore::PVSerializeObject;
public:
	PVPlottingProperties(PVRush::PVFormat const& fmt, int idx);
	PVPlottingFilter::p_type get_plotting_filter() { return _plotting_filter; };

protected:
	// Serialization
	PVPlottingProperties() { }

public:
	void set_mode(QString const& mode);

public:
	bool operator==(PVPlottingProperties const& org);

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

private:
	QString _type;
	QString _mode;
	PVCol _index;
	PVPlottingFilter::p_type _plotting_filter;
};
}

#endif	/* PICVIZ_PVPLOTTINGPROPERTIES_H */
