#ifndef PICVIZ_PVSELROWFILTERINGFUNCTION_H
#define PICVIZ_PVSELROWFILTERINGFUNCTION_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVBinaryOperation.h>
#include <pvkernel/core/PVFunctionArgs.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVCompList.h>

#include <picviz/PVSelRowFilteringFunction_types.h>

#include <boost/enable_shared_from_this.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

namespace Picviz {

class PVSelection;
class PVSparseSelection;
class PVView;

/*! \brief Interface from selection row filtering functions
 */
class PVSelRowFilteringFunction: public PVCore::PVFunctionArgs<boost::function<void(PVRow, PVView const&, PVView const&, PVSparseSelection&)> >, public PVCore::PVRegistrableClass<PVSelRowFilteringFunction>, public boost::enable_shared_from_this<PVSelRowFilteringFunction>
{
	typedef PVCore::PVFunctionArgs<boost::function<void(PVRow, PVView const&, PVView const&, PVSparseSelection&)> > fargs_t;
	typedef PVCore::PVRegistrableClass<PVSelRowFilteringFunction> reg_class_t;
public:
	typedef boost::shared_ptr<PVSelRowFilteringFunction> p_type;
public:
	PVSelRowFilteringFunction():
		fargs_t(),
		reg_class_t(),
		_do_pre_process(true),
		_combination_op(PVCore::PVBinaryOperation::OR)
	{ }

public:
	virtual void set_args(PVCore::PVArgumentList const& args)
	{
		_do_pre_process = !PVCore::comp_hash(_last_args, args, get_args_keys_for_preprocessing());
		PVCore::PVFunctionArgsBase::set_args(args);
	}
	void pre_process(PVView const& view_org, PVView const& view_dst)
	{
		if (_do_pre_process) {
			do_pre_process(view_org, view_dst);
			_last_args = _args;
			_do_pre_process = false ;
		}
	}
	virtual void operator()(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSparseSelection& sel_dst) const = 0;

	// Optimised version for OR-only ops
	virtual void process_or(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSelection& sel_dst) const = 0;

	virtual QString get_human_name() const { return registered_name(); }
	virtual QString get_human_name_with_args(const PVView& /*src_view*/, const PVView& /*dst_view*/) const { return get_human_name(); }

public:
	inline PVCore::PVArgumentList get_args_for_org_view() const { return get_args_for_view(get_arg_keys_for_org_view()); }
	inline PVCore::PVArgumentList get_args_for_dst_view() const { return get_args_for_view(get_arg_keys_for_dst_view()); }
	inline PVCore::PVArgumentList get_global_args() const { return get_args_for_view(get_global_arg_keys()); }

	inline void set_args_for_org_view(PVCore::PVArgumentList const& v_args) { set_args_for_view(v_args, get_arg_keys_for_org_view()); }
	inline void set_args_for_dst_view(PVCore::PVArgumentList const& v_args) { set_args_for_view(v_args, get_arg_keys_for_dst_view()); }

	inline PVCore::PVBinaryOperation get_combination_op() const { return _combination_op; }
	inline void set_combination_op(PVCore::PVBinaryOperation op) { _combination_op = op; }

public:
	void to_xml(QDomElement& elt) const;
	static p_type from_xml(QDomElement const& elt);

protected:
	virtual void do_pre_process(PVView const& view_org, PVView const& view_dst) = 0;
	virtual PVCore::PVArgumentKeyList get_args_keys_for_preprocessing() { return get_default_args().keys(); }
	virtual PVCore::PVArgumentKeyList get_arg_keys_for_org_view() const = 0;
	virtual PVCore::PVArgumentKeyList get_arg_keys_for_dst_view() const = 0;
	virtual PVCore::PVArgumentKeyList get_global_arg_keys() const
	{
		PVCore::PVArgumentList args = _args;
		foreach (PVCore::PVArgumentKey key, get_arg_keys_for_org_view()) {
			args.remove(key);
		}
		foreach (PVCore::PVArgumentKey key, get_arg_keys_for_dst_view()) {
			args.remove(key);
		}
		return args.keys();
	}

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

	bool _do_pre_process;
	PVCore::PVArgumentList _last_args;
	PVCore::PVBinaryOperation _combination_op;
};

#define CLASS_RFF(T)\
	public:\
		virtual func_type f() { return boost::bind<void>((void(T::*)(PVRow, PVView const&, PVView const&, PVSparseSelection&))(&T::operator()), this, _1, _2, _3, _4); }\
	CLASS_FUNC_ARGS_PARAM(T) \
	CLASS_REGISTRABLE(T)

#define CLASS_RFF_NOPARAM(T)\
	public:\
		virtual func_type f() { return boost::bind<void>((void(T::*)(PVRow, PVView const&, PVView const&, PVSparseSelection&))(&T::operator()), this, _1, _2, _3, _4); }\
	CLASS_FUNC_ARGS_NOPARAM(T)\
	CLASS_REGISTRABLE(T)


}

#endif
