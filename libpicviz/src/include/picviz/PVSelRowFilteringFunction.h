#ifndef PICVIZ_PVSELROWFILTERINGFUNCTION_H
#define PICVIZ_PVSELROWFILTERINGFUNCTION_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <picviz/PVSelection.h>
#include <picviz/PVSelRowFilteringFunction_types.h>

namespace Picviz {

class PVView;

/*! \brief Interface from selection row filtering functions
 */
class PVSelRowFilteringFunction: public PVCore::PVFunctionArgs<boost::function<void(PVRow, PVView const&, PVView const&, PVSelection&)> >, public PVCore::PVRegistrableClass<PVSelRowFilteringFunction>
{
public:
	typedef boost::shared_ptr<PVSelRowFilteringFunction> p_type;
	typedef PVSelRowFilteringFunction base_registrable;
public:
	virtual void pre_process(PVView const& /*view_org*/, PVView const& /*view_dst*/) { }
	virtual void operator()(PVRow /*row_org*/, PVView const& /*view_org*/, PVView const& /*view_dst*/, PVSelection& /*sel_dst*/) const { }
	
public:
	inline PVCore::PVArgumentList get_args_for_org_view() const { return get_args_for_view(get_arg_keys_for_org_view()); }
	inline PVCore::PVArgumentList get_args_for_dst_view() const { return get_args_for_view(get_arg_keys_for_dst_view()); }

	inline void set_args_for_org_view(PVCore::PVArgumentList const& v_args) { set_args_for_view(v_args, get_arg_keys_for_org_view()); }
	inline void set_args_for_dst_view(PVCore::PVArgumentList const& v_args) { set_args_for_view(v_args, get_arg_keys_for_dst_view()); }

	func_type f() { return boost::bind<void>((void(PVSelRowFilteringFunction::*)(PVRow, PVView const&, PVView const&, PVSelection&))(&PVSelRowFilteringFunction::operator()), this, _1, _2, _3, _4); }

protected:
	virtual PVCore::PVArgumentKeyList get_arg_keys_for_org_view() const { return get_default_args().keys(); };
	virtual PVCore::PVArgumentKeyList get_arg_keys_for_dst_view() const { return get_default_args().keys(); };

private:
	inline PVCore::PVArgumentList get_args_for_view(PVCore::PVArgumentKeyList const& keys) const
	{
		return PVCore::filter_argument_list_with_keys(get_args(), keys, get_default_args());
	}

	inline void set_args_for_view(PVCore::PVArgumentList const& v_args, PVCore::PVArgumentKeyList const& keys)
	{
		PVCore::PVArgumentList args = get_args();
		PVCore::PVArgumentList_set_common_args_from(args, filter_argument_list_with_keys(v_args, keys, get_default_args()));
		set_args(args);
	}
};

#define CLASS_RFF(T)\
	public:\
		virtual func_type f() { return boost::bind<void>((void(T::*)(PVRow, PVView const&, PVView const&, PVSelection&))(&T::operator()), this, _1, _2, _3, _4); }\
	CLASS_FUNC_ARGS_PARAM(T) \
	CLASS_REGISTRABLE(T)

#define CLASS_RFF_NOPARAM(T)\
	public:\
		virtual func_type f() { return boost::bind<void>((void(T::*)(PVRow, PVView const&, PVView const&, PVSelection&))(&T::operator()), this, _1, _2, _3, _4); }\
	CLASS_FUNC_ARGS_NOPARAM(T)\
	CLASS_REGISTRABLE(T)


}

#endif
