/**
 * \file PVLayerFilterFindDuplicates.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVLAYERFILTERFindDuplicates_H
#define PICVIZ_PVLAYERFILTERFindDuplicates_H


#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterFindDuplicates
 */
class PVLayerFilterFindDuplicates : public PVLayerFilter {
public:
	PVLayerFilterFindDuplicates(PVCore::PVArgumentList const& l = PVLayerFilterFindDuplicates::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;
	virtual QString menu_name() const { return "Find/Duplicates"; }

public:
	CLASS_FILTER(Picviz::PVLayerFilterFindDuplicates)
};
}

#endif
