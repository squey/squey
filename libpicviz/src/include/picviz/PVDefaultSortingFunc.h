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
	virtual fequals_type f_equals();
	virtual fless_type f_less();

private:
	static bool less_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool less_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	static bool equals_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool equals_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	static int comp_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static int comp_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	CLASS_REGISTRABLE(PVDefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVDefaultSortingFunc)
};

}

#endif
