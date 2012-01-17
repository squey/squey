#ifndef PICVIZ_PVSORTINGFUNC_H
#define PICVIZ_PVSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVFunctionArgs.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <boost/shared_ptr.hpp>

namespace PVCore {
class PVUnicodeString;
}

namespace Picviz {

namespace __impl {
	typedef bool(*unicode_sorting_comp_func)(PVCore::PVUnicodeString const&, PVCore::PVUnicodeString const&);
}

class LibPicvizDecl PVSortingFunc : public PVCore::PVFunctionArgs<__impl::unicode_sorting_comp_func>, public PVCore::PVRegistrableClass<PVSortingFunc>
{
public:
	typedef boost::shared_ptr<PVSortingFunc> p_type;
	typedef __impl::unicode_sorting_comp_func f_type;

public:
	PVSortingFunc(PVCore::PVArgumentList const& l = PVSortingFunc::default_args());

public:
	virtual f_type f() = 0;

	CLASS_FUNC_ARGS_NOPARAM()
};

typedef PVSortingFunc::p_type PVSortingFunc_p;
typedef PVSortingFunc::f_type PVSortingFunc_f;

}

#endif
