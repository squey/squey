/**
 * \file PVLayerFilterHeatline.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);

protected:
	virtual void post(const PVLayer &in, PVLayer &out,
	                  const double ratio, const double fmin, const double fmax,
	                  const PVRow line_id);

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineBase)
};

class PVLayerFilterHeatlineColor : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineColor(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineColor::default_args());
protected:
	void post(const PVLayer &in, PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id) override;

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineColor)
};

class PVLayerFilterHeatlineSel : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSel(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineSel::default_args());
protected:
	void post(const PVLayer &in, PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id) override;

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineSel)
};

class PVLayerFilterHeatlineSelAndCol : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSelAndCol(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineSelAndCol::default_args());
protected:
	void post(const PVLayer &in, PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id) override;

	virtual QString menu_name() const { return "Frequency gradient"; }

	CLASS_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
};

}

#endif
