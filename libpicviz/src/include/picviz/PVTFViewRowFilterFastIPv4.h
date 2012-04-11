#ifndef PICVIZ_PVTFVIEWROWFILTERFASTIPV4_H
#define PICVIZ_PVTFVIEWROWFILTERFASTIPV4_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>
#include <picviz/PVTransformationFunctionView.h>
#include <picviz/PVSelRowFilteringFunction_types.h>

#include <QList>

namespace Picviz {

class PVView;

class LibPicvizDecl PVTFViewRowFilterFastIPv4: public PVTransformationFunctionView
{
public:
	typedef QList<PVSelRowFilteringFunction_p> list_rff_t;

public:
	PVSelection operator()(PVView const& view_src, PVView const& view_dst, PVSelection const& sel_org) const;

public:
	void push_rff(PVSelRowFilteringFunction_p rff) { _rffs << rff; }

protected:
	list_rff_t _rffs;
};

}

#endif
