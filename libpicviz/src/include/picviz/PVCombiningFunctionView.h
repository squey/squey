#ifndef PICVIZ_PVCOMBININGFUNCTIONVIEW_H
#define PICVIZ_PVCOMBININGFUNCTIONVIEW_H

#include <picviz/PVView.h>
#include <picviz/PVSelection.h>

namespace Picviz {

class PVCombiningFunctionView
{
public:
	PVCombiningFunctionView() {}
	~PVCombiningFunctionView() {}

public:
	PVSelection operator() (const PVView &va, const PVView &vb) { return PVSelection(); }
};

}

#endif // PICVIZ_PVCOMBININGFUNCTIONVIEW_H
