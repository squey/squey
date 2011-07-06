//! \file PVLayerFilterHeatline.h
//! $Id: PVLayerFilterHeatline.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERHeatline_H
#define PICVIZ_PVLAYERFILTERHeatline_H


#include <pvcore/general.h>
#include <pvcore/types.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterHeatline
 */
class LibExport PVLayerFilterHeatlineBase : public PVLayerFilter {
public:
	PVLayerFilterHeatlineBase(PVFilter::PVArgumentList const& l = PVLayerFilterHeatlineBase::default_args());
public:
	void operator()(PVLayer& in, PVLayer &out);
	PVFilter::PVArgumentList get_default_args_for_view(PVView const& view);
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineBase)

};

class LibExport PVLayerFilterHeatlineColor : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineColor(PVFilter::PVArgumentList const& l = PVLayerFilterHeatlineColor::default_args());
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineColor)
};

class LibExport PVLayerFilterHeatlineSel : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSel(PVFilter::PVArgumentList const& l = PVLayerFilterHeatlineSel::default_args());
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineSel)
};

class LibExport PVLayerFilterHeatlineSelAndCol : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSelAndCol(PVFilter::PVArgumentList const& l = PVLayerFilterHeatlineSelAndCol::default_args());
protected:
	virtual void post(PVLayer &in, PVLayer &out, float ratio, PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
};

}

#endif
