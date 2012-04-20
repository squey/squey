#ifndef PICVIZ_PVCOMBININGFUNCTIONVIEW_H
#define PICVIZ_PVCOMBININGFUNCTIONVIEW_H

#include <picviz/PVView.h>
#include <picviz/PVSelection.h>
#include <picviz/PVTransformationFunctionView_types.h>

#include <QList>

namespace Picviz {

class PVTFViewRowFiltering;

class LibPicvizDecl PVCombiningFunctionView
{
private:
	typedef QList<PVTransformationFunctionView_p> list_tf_t;
public:
	PVCombiningFunctionView();
	virtual ~PVCombiningFunctionView() {}

public:
	void pre_process(const PVView &view_src, const PVView &view_dst);
	PVSelection operator() (const PVView &view_src, const PVView &view_dst) const;
	PVTFViewRowFiltering* get_first_tf();

protected:
	list_tf_t _tfs;
};

}

#endif // PICVIZ_PVCOMBININGFUNCTIONVIEW_H
