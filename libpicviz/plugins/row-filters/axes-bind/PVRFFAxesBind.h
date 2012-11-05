/**
 * \file PVRFFAxesBind.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVRFFAXESBIND_H
#define PICVIZ_PVRFFAXESBIND_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVSparseSelection.h>
#include <picviz/PVMapped_types.h>

#include <unordered_map>

//#define NTHREADS 2

namespace Picviz {

class LibPicvizDecl PVRFFAxesBind: public PVSelRowFilteringFunction
{
private:
	typedef std::unordered_map<Picviz::mapped_decimal_storage_type, std::vector<PVRow> > hash_rows;
	//typedef boost::unordered_map<float, Picviz::PVSparseSelection> hash_rows;
public:
	PVRFFAxesBind(PVCore::PVArgumentList const& l = PVRFFAxesBind::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	virtual QString get_human_name() const { return "Axes Bind"; }
	virtual QString get_human_name_with_args(const PVView& src_view, const PVView& dst_view) const;

	void do_pre_process(PVView const& view_org, PVView const& view_dst);
	void operator()(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSparseSelection& sel_dst) const;
	void process_or(PVRow row_org, PVView const& view_org, PVView const& view_dst, PVSelection& sel_dst) const;

protected:
	PVCore::PVArgumentKeyList get_arg_keys_for_org_view() const { return PVCore::PVArgumentKeyList() << "axis_org"; }
	PVCore::PVArgumentKeyList get_arg_keys_for_dst_view() const { return PVCore::PVArgumentKeyList() << "axis_dst"; }

private:
	PVCol _axis_org;
	PVCol _axis_dst;

	//hash_rows _dst_values[NTHREADS];
	hash_rows _dst_values;

	CLASS_RFF(PVRFFAxesBind);
};

}

#endif
