//! \file PVPlottingProperties.h
//! $Id: PVPlottingProperties.h 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVPLOTTINGPROPERTIES_H
#define PICVIZ_PVPLOTTINGPROPERTIES_H

//#include <QList>

#include <pvcore/general.h>
#include <pvrush/PVFormat.h>

//#include <picviz/function.h>

#include <picviz/PVPlottingFunction.h>
#include <picviz/PVRoot.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

/**
* \class PVPlottingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class LibExport PVPlottingProperties {
private:
	PVRush::PVFormat format;
	PVCol index;
public:
	PVPlottingProperties(PVRoot_p root, PVRush::PVFormat fmt, int idx);
	PVPlottingFilter::p_type plotting_filter;
};
}

#endif	/* PICVIZ_PVPLOTTINGPROPERTIES_H */
