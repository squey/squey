//! \file PVLayerFilterCreatelayers.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERCreateLayers_H
#define PICVIZ_PVLAYERFILTERCreateLayers_H

#include <pvkernel/core/general.h>

#include <QList>
#include <QString>
#include <QPair>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterCreateLayers
 */
class PVLayerFilterCreateLayers : public PVLayerFilter {
private:
	QMap<QString, QList<QPair<QString, QString> > > create_layers_config;
	QString _menu_name;

public:
	PVLayerFilterCreateLayers(QString menu_name,PVCore::PVArgumentList const& l = PVLayerFilterCreateLayers::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
	int create_layers_get_config(QString filename);

	CLASS_FILTER(Picviz::PVLayerFilterCreateLayers)

};
}

#endif
