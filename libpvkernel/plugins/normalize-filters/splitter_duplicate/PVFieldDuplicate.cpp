/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldDuplicate.h"

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::PVFieldDuplicate
 *
 *****************************************************************************/
PVFilter::PVFieldDuplicate::PVFieldDuplicate(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldDuplicate, args);
}

void PVFilter::PVFieldDuplicate::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_n = std::max((uint32_t) args.at("n").toUInt(), (uint32_t) 2);
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldDuplicate)
{
	PVCore::PVArgumentList args;
	args["n"] = 2;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldDuplicate::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	PVCore::list_fields::size_type ret = 0;

	for (size_t i = 0; i < _n; i++) {
		PVCore::PVField &ins_f(*l.insert(it_ins, field));
		ins_f.allocate_new(field.size());
		memcpy(ins_f.begin(), field.begin(), field.size());
		ins_f.set_end(ins_f.begin() + field.size());
		ret++;
	}

	return ret;
}


IMPL_FILTER(PVFilter::PVFieldDuplicate)
