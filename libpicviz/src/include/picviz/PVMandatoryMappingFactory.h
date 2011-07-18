//! \file PVMandatoryMappingFactory.h
//! $Id: PVMandatoryMappingFactory.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMANDATORYMAPPINGFACTORY_H
#define PICVIZ_PVMANDATORYMAPPINGFACTORY_H

#include <QHash>
#include <QList>
#include <QString>
#include <QVector>

#include <pvcore/general.h>
#include <picviz/PVMandatoryMappingFunction.h>

namespace Picviz {

/**
 * \class PVMandatoryMappingFactory
 */
class LibPicvizDecl PVMandatoryMappingFactory {
public:
	PVMandatoryMappingFactory();
	~PVMandatoryMappingFactory();

	int register_all();
	QVector<Picviz::PVMandatoryMappingFunction *> plugins;
};
}

#endif	/* PICVIZ_PVMANDATORYMAPPINGFACTORY_H */
