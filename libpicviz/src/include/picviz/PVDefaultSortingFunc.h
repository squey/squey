#ifndef PICVIZ_PVDEFAULTSORTINGFUNC_H
#define PICVIZ_PVDEFAULTSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <picviz/PVSortingFunc.h>

namespace Picviz {

class LibPicvizDecl PVDefaultSortingFunc : public PVSortingFunc
{
public:
	PVDefaultSortingFunc(PVCore::PVArgumentList const& l = PVDefaultSortingFunc::default_args());
public:
	virtual f_type f();

private:
	static bool comp_match_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool comp_match_desc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	CLASS_REGISTRABLE(PVDefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVDefaultSortingFunc)
};

}

#endif
