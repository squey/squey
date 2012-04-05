#ifndef PICVIZ_PVRFFAXESBIND_H
#define PICVIZ_PVRFFAXESBIND_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>

namespace Picviz {

class LibPicvizDecl PVRFFAxesBind: public PVSelRowFilteringFunction
{
public:
	PVRFFAxesBind(PVCore::PVArgumentList const& l = PVRFFAxesBind::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	void operator()(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSelection& sel_dst) const;

private:
	PVCol _axis_org;
	PVCol _axis_dst;

	CLASS_RFF(PVRFFAxesBind);
};

}

#endif
