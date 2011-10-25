//! \file PVMappingProperties.h
//! $Id: PVMappingProperties.h 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMAPPINGPROPERTIES_H
#define PICVIZ_PVMAPPINGPROPERTIES_H

//#include <QList>

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVMappingFunction.h>
#include <picviz/PVRoot.h>

#include <picviz/PVMappingFilter.h>

namespace Picviz {

/**
* \class PVMappingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class LibPicvizDecl PVMappingProperties {
private:
public:
	PVMappingProperties(PVMapping const& parent, PVRush::PVFormat const& fmt, int idx);
	QString get_group_key() const { return _group_key; }
	void set_mode(QString const& mode);
	inline PVMappingFilter::p_type get_mapping_filter() const { assert(_mapping_filter); return _mapping_filter; }

public:
	bool operator==(const PVMappingProperties& org);

private:
	PVCol _index;
	QString _group_key;
	PVSource const* _src_parent;
	PVMappingFilter::p_type _mapping_filter;
	QString _type;
};

}

#endif	/* PICVIZ_PVMAPPINGPROPERTIES_H */
