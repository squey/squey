#ifndef PICVIZ_PVRFFAXESBIND_H
#define PICVIZ_PVRFFAXESBIND_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>

#include <boost/unordered_map.hpp>
#include <map>

namespace Picviz {

class LibPicvizDecl PVRFFAxesBindNearestNeighbors: public PVSelRowFilteringFunction
{
private:
	typedef std::map<float,std::vector<PVRow>, std::less<float> > map_rows;

public:
	PVRFFAxesBindNearestNeighbors(PVCore::PVArgumentList const& l = PVRFFAxesBindNearestNeighbors::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	void pre_process(PVView const& view_org, PVView const& view_dst);
	void operator()(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSelection& sel_dst) const;

protected:
	PVCore::PVArgumentKeyList get_arg_keys_for_org_view() const { return PVCore::PVArgumentKeyList() << "axis_org"; }
	PVCore::PVArgumentKeyList get_arg_keys_for_dst_view() const { return PVCore::PVArgumentKeyList() << "axis_dst"; }

private:
	PVCol _axis_org;
	PVCol _axis_dst;
	float _distance;

	map_rows _dst_values;

	CLASS_RFF(PVRFFAxesBindNearestNeighbors);
};

}

#endif
