/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERFILTERHeatline_H
#define INENDI_PVLAYERFILTERHeatline_H


#include <pvkernel/core/general.h>
#include <pvbase/types.h>

#include <inendi/PVLayer.h>
#include <inendi/PVLayerFilter.h>

namespace Inendi {

/**
 * \class PVLayerFilterHeatline
 */
class PVLayerFilterHeatlineBase : public PVLayerFilter {
public:
	PVLayerFilterHeatlineBase(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineBase::default_args());
public:
	void operator()(PVLayer& in, PVLayer &out);
	virtual PVCore::PVArgumentKeyList get_args_keys_for_preset() const;
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);

protected:
	virtual void post(const PVLayer &in, PVLayer &out,
	                  const double ratio, const double fmin, const double fmax,
	                  const PVRow line_id);

	CLASS_FILTER(Inendi::PVLayerFilterHeatlineBase)
};

class PVLayerFilterHeatlineColor : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineColor(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineColor::default_args());
protected:
	void post(const PVLayer &in, PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id) override;

	CLASS_FILTER(Inendi::PVLayerFilterHeatlineColor)
};

class PVLayerFilterHeatlineSel : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSel(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineSel::default_args());
protected:
	void post(const PVLayer &in, PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id) override;

	CLASS_FILTER(Inendi::PVLayerFilterHeatlineSel)
};

class PVLayerFilterHeatlineSelAndCol : public PVLayerFilterHeatlineBase {
public:
	PVLayerFilterHeatlineSelAndCol(PVCore::PVArgumentList const& l = PVLayerFilterHeatlineSelAndCol::default_args());
protected:
	void post(const PVLayer &in, PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id) override;

	virtual QString menu_name() const { return "Frequency gradient"; }

	CLASS_FILTER(Inendi::PVLayerFilterHeatlineSelAndCol)
};

}

#endif
