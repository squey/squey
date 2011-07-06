//! \file PVMappingProperties.h
//! $Id: PVMappingProperties.h 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMAPPINGPROPERTIES_H
#define PICVIZ_PVMAPPINGPROPERTIES_H

//#include <QList>

#include <pvcore/general.h>
#include <pvrush/PVFormat.h>

//#include <picviz/function.h>

#include <picviz/PVMappingFunction.h>
#include <picviz/PVRoot.h>

#include <picviz/PVMappingFilter.h>

namespace Picviz {

/**
* \class PVMappingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class LibExport PVMappingProperties {
private:
	PVRush::PVFormat format;
	PVCol index;
public:
	PVMappingProperties(PVRoot_p root, PVRush::PVFormat fmt, int idx);
	PVMappingFilter::p_type mapping_filter;
};
}

#endif	/* PICVIZ_PVMAPPINGPROPERTIES_H */
