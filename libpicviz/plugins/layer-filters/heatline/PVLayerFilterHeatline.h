//! \file PVLayerFilterHeatline.h
//! $Id: PVLayerFilterHeatline.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERHeatline_H
#define PICVIZ_PVLAYERFILTERHeatline_H


#include <pvkernel/core/general.h>
#include <pvbase/types.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterHeatline
 */
class PVLayerFilterHeatlineBase : public PVLayerFilter {
public:
	PVLayerFilterHeatlineBase(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineBase::default_args());
public:
	void operator()(PVLayer& in, PVLayer &out);
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineBase)

};

class PVLayerFilterHeatlineColor : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineColor(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineColor::default_args());
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineColor)
};

class PVLayerFilterHeatlineSel : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSel(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineSel::default_args());
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineSel)
};

class PVLayerFilterHeatlineSelAndCol : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSelAndCol(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineSelAndCol::default_args());
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
};

}

#endif
