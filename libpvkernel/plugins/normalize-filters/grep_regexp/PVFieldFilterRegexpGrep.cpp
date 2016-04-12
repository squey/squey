/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVFieldFilterRegexpGrep.h"


/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrepg::PVFieldFilterGrep
 *
 *****************************************************************************/
PVFilter::PVFieldFilterRegexpGrep::PVFieldFilterRegexpGrep(PVCore::PVArgumentList const& args) :
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>()
{
	INIT_FILTER(PVFilter::PVFieldFilterRegexpGrep, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
{
	PVCore::PVArgumentList args;
	args["regexp"] = QString("");
	args["reverse"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrep::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldFilterRegexpGrep::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_rx = std::regex(args.at("regexp").toString().toStdString());
	_inverse = args.at("reverse").toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrep::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldFilterRegexpGrep::one_to_one(PVCore::PVField& field)
{
	std::cmatch base_match;
	bool found = std::regex_search<const char*>(field.begin(), field.end(), base_match, _rx);
	found |= base_match.size() > 1;
	if (not (found ^ _inverse)) {
		field.set_invalid();
	}
	return field;
}

IMPL_FILTER(PVFilter::PVFieldFilterRegexpGrep)
