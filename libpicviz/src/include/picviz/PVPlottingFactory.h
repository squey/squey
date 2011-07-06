//! \file PVPlottingFactory.h
//! $Id: PVPlottingFactory.h 2590 2011-05-07 15:43:12Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVPLOTTINGFACTORY_H
#define PICVIZ_PVPLOTTINGFACTORY_H

#include <QHash>
#include <QList>
#include <QString>

#include <pvcore/general.h>
#include <picviz/PVPlottingFunction.h>


namespace Picviz {

/**
 * \class PVPlottingFactory
 */
class LibExport PVPlottingFactory {
public:
	PVPlottingFactory();
	~PVPlottingFactory();

	int register_all();
	QHash<QString, QHash<QString, PVPlottingFunction * > > plugins;
	/*           "enum"       "default"  */
};
}

#endif	/* PICVIZ_PVPLOTTINGFACTORY_H */
