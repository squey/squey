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
	float* operator()(float* value);
	void init_expand(float min, float max);
	float expand_plotted(float value) const;
	QString get_human_name() const { return QString("Logarithmic min/max"); }
	bool can_expand() const { return true; }

private:
	float _expand_min;
	float _expand_max;
	float _expand_diff;
	float _offset;

	CLASS_FILTER(PVPlottingFilterLogMinmax)
};

}

#endif
