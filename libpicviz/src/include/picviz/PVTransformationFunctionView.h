#ifndef PICVIZ_PVTRANSFORMATIONFUNCTIONVIEW_H
#define PICVIZ_PVTRANSFORMATIONFUNCTIONVIEW_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>

namespace Picviz {

class PVView;

/*! \brief Interface for view's selection transformation
 */
class PVTransformationFunctionView
{
public:
	PVTransformationFunctionView() { }

public:
	virtual PVSelection operator()(PVView const& view_org, PVView const& view_dst, PVSelection const& sel_org) const = 0;
};

}

#endif
