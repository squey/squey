#ifndef PICVIZ_PVRFFAXESBIND_H
#define PICVIZ_PVRFFAXESBIND_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>

#include <boost/unordered_map.hpp>

//#define NTHREADS 2

namespace Picviz {

class LibPicvizDecl PVRFFAxesBind: public PVSelRowFilteringFunction
{
private:
	typedef boost::unordered_map<float,std::vector<PVRow> > hash_rows;
public:
	PVRFFAxesBind(PVCore::PVArgumentList const& l = PVRFFAxesBind::default_args());

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

	//hash_rows _dst_values[NTHREADS];
	hash_rows _dst_values;

	CLASS_RFF(PVRFFAxesBind);
};

}

#endif
