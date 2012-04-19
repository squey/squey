#ifndef PICVIZ_PVTFVIEWROWFILTERING_H
#define PICVIZ_PVTFVIEWROWFILTERING_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>
#include <picviz/PVTransformationFunctionView.h>
#include <picviz/PVSelRowFilteringFunction_types.h>

#include <QList>

namespace Picviz {

class PVView;

class LibPicvizDecl PVTFViewRowFiltering: public PVTransformationFunctionView
{
public:
	typedef QList<PVSelRowFilteringFunction_p> list_rff_t;

public:
	void pre_process(PVView const& view_src, PVView const& view_dst);

	PVSelection operator()(PVView const& view_src, PVView const& view_dst, PVSelection const& sel_org) const;
	list_rff_t get_rffs() { return _rffs; }

public:
	void push_rff(PVSelRowFilteringFunction_p rff) { _rffs << rff; }
	void remove_rff(int index) { _rffs.removeAt(index); }
	bool remove_rff(PVSelRowFilteringFunction_p rff) { return _rffs.removeOne(rff); }

protected:
	list_rff_t _rffs;
};

}

#endif
