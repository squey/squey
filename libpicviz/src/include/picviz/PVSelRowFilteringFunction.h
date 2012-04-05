#ifndef PICVIZ_PVSELROWFILTERINGFUNCTION_H
#define PICVIZ_PVSELROWFILTERINGFUNCTION_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>

namespace Picviz {

class PVView;

/*! \brief Interface from selection row filtering functions
 */
class PVSelRowFilteringFunction: public PVCore::PVFunctionArgs<boost::function<void(PVRow, PVView const&, PVView const&, PVSelection&)> >
{
public:
	virtual void operator()(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSelection& sel_dst) const = 0;
};

#define CLASS_RFF(T)\
	public:\
		virtual func_type f() { return boost::bind<void>((void(T::*)(PVRow, PVView const&, PVView const&, PVSelection&))(&T::operator()), this, _1, _2, _3, _4); }\
	CLASS_FUNC_ARGS_PARAM(T)

#define CLASS_RFF_NOPARAM(T)\
	public:\
		virtual func_type f() { return boost::bind<void>((void(T::*)(PVRow, PVView const&, PVView const&, PVSelection&))(&T::operator()), this, _1, _2, _3, _4); }\
	CLASS_FUNC_ARGS_NOPARAM(T)

}

#endif
