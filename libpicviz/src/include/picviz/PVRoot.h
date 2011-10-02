//! \file PVRoot.h
//! $Id: PVRoot.h 3068 2011-06-07 11:13:39Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVROOT_H
#define PICVIZ_PVROOT_H

#include <QList>
#include <QStringList>

#include <pvkernel/core/general.h>

#include <picviz/PVMandatoryMappingFactory.h>
#include <picviz/PVMappingFactory.h>
#include <picviz/PVPlottingFactory.h>

#include <picviz/PVPtrObjects.h> // For PVScene_p

#include <boost/shared_ptr.hpp>

// Plugins prefix
#define LAYER_FILTER_PREFIX "layer_filter"
#define MAPPING_FILTER_PREFIX "mapping_filter"
#define PLOTTING_FILTER_PREFIX "plotting_filter"

namespace Picviz {

/**
 * \class PVRoot
 */
class LibPicvizDecl PVRoot {
public:
	typedef boost::shared_ptr<PVRoot> p_type;
public:
	PVRoot();
	~PVRoot();

	/* Properties */
	QList<PVScene_p> scenes;

	/* Functions */
	int scene_append(PVScene_p scene);
private:
	static QStringList split_plugin_dirs(QString const& dirs);

	// Plugins loading
	static int load_layer_filters();
	static int load_mapping_filters();
	static int load_plotting_filters();
};

typedef PVRoot::p_type PVRoot_p;

}

#endif	/* PICVIZ_PVROOT_H */
