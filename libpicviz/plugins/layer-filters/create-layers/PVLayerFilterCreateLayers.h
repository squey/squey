/**
 * \file PVLayerFilterCreateLayers.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
	QString _section_name;
	QMap <QString, QStringList> _layers_regex;

public:
	PVLayerFilterCreateLayers(QString section_name, QMap<QString, QStringList> layers_regex, PVCore::PVArgumentList const& l = PVLayerFilterCreateLayers::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);

	CLASS_FILTER(Picviz::PVLayerFilterCreateLayers)

};
}

#endif
