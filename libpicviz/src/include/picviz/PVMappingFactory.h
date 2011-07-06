//! \file PVMappingFactory.h
//! $Id: PVMappingFactory.h 2590 2011-05-07 15:43:12Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMAPPINGFACTORY_H
#define PICVIZ_PVMAPPINGFACTORY_H

#include <QHash>
#include <QList>
#include <QString>

#include <pvcore/general.h>
#include <picviz/PVMappingFunction.h>


namespace Picviz {

/**
 * \class PVMappingFactory
 */
class LibExport PVMappingFactory {
public:
	PVMappingFactory();
	~PVMappingFactory();

	int register_all();
	QHash<QString, QHash<QString, PVMappingFunction * > > plugins;
	/*           "enum"       "default"  */
};
}

#endif	/* PICVIZ_PVMAPPINGFACTORY_H */
