/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldSplitterRegexp.h"
#include <pvkernel/core/PVBufferSlice.h>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldSplitterRegexp, args);
}

PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp(const PVFieldSplitterRegexp& src) :
	PVFieldsFilter<PVFilter::one_to_many>(src)
{
	_regexp = src._regexp;
	_full_line = src._full_line;
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
{
	PVCore::PVArgumentList args;
	args["regexp"] = PVCore::PVArgument(QString(""));
	args["full-line"] = PVCore::PVArgument(true);
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterRegexp::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_regexp = std::regex(args.at("regexp").toString().toStdString());
	_full_line = args.at("full-line").toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::one_to_many
 *
 *****************************************************************************/

PVCore::list_fields::size_type PVFilter::PVFieldSplitterRegexp::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	std::cmatch base_match;
	bool parse_success = true;
	if(_full_line) {
		parse_success = std::regex_match<const char*>(field.begin(), field.end(), base_match, _regexp);
	} else {
		parse_success = std::regex_search<const char*>(field.begin(), field.end(), base_match, _regexp);
	}

	if (parse_success) {
		if((_fields_expected > 0) && (_fields_expected != base_match.size() - 1)) {
			field.set_invalid();
			field.elt_parent()->set_invalid();
			return 0;
		}
		for(auto it = ++base_match.begin(); it != base_match.end(); it++) {
			PVCore::list_fields::value_type elt(field);
			elt.set_begin(field.begin() + std::distance(static_cast<const char*>(field.begin()), it->first));
			elt.set_end(field.begin() + std::distance(static_cast<const char*>(field.begin()), it->second));
			elt.set_physical_end(field.begin() + std::distance(static_cast<const char*>(field.begin()), it->second));
			l.insert(it_ins, elt);
		}

		return base_match.size() - 1;
	}

	return 0;
}

IMPL_FILTER(PVFilter::PVFieldSplitterRegexp)
