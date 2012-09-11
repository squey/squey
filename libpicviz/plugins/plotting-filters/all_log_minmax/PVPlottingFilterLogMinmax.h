/**
 * \file PVPlottingFilterLogMinmax.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVPLOTTINGFILTERLOGMINMAX_H
#define PVFILTER_PVPLOTTINGFILTERLOGMINMAX_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterLogMinmax: public PVPlottingFilter
{
public:
	PVPlottingFilterLogMinmax(PVCore::PVArgumentList const& args = PVPlottingFilterLogMinmax::default_args());

public:
	uint32_t* operator()(mapped_decimal_storage_type const* value) override;
	void init_expand(uint32_t min, uint32_t max) override;
	uint32_t expand_plotted(uint32_t value) const override;
	QString get_human_name() const override { return QString("Logarithmic min/max"); }
	bool can_expand() const { return true; }

private:
	double _expand_min;
	double _expand_max;
	double _expand_diff;
	uint32_t _offset;

	CLASS_FILTER(PVPlottingFilterLogMinmax)
};

}

#endif
