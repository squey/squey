/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldSplitterIP.h"

#include <numeric>
#include <iostream>

const QString PVFilter::PVFieldSplitterIP::sep = QString(",");

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIP::PVFieldSplitterIP
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterIP::PVFieldSplitterIP(PVCore::PVArgumentList const& args)
    : PVFieldsSplitter()
{
	INIT_FILTER(PVFilter::PVFieldSplitterIP, args);
}

void PVFilter::PVFieldSplitterIP::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_ipv6 = args.at("ipv6").toBool();
	QString params = args.at("params").toString();

	// Compute adjacente difference to have "number of elements to search" from current position.
	_indexes.clear();
	for (const QString& s : params.split(sep, QString::SkipEmptyParts)) {
		_indexes.push_back(s.toUInt());
	}
	std::sort(_indexes.begin(), _indexes.end());
	std::unique(_indexes.begin(), _indexes.end());
	std::adjacent_difference(_indexes.begin(), _indexes.end(), _indexes.begin());
	_indexes[0]++;
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterIP)
{
	PVCore::PVArgumentList args;
	args["ipv6"] = false;
	args["params"] = "0,1,2";
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIP::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterIP::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	char token = (_ipv6) ? ':' : '.';

	char* pos = field.begin();
	PVCore::PVField f(field);
	for (size_t index : _indexes) {
		f.set_begin(pos);
		for (size_t j = 0; j < index; j++) {
			// + 1 to be after the token
			pos = std::find(pos, field.end(), token) + 1;
		}
		f.set_end(pos - 1);
		l.insert(it_ins, f);
	}

	return _indexes.size();
}

IMPL_FILTER(PVFilter::PVFieldSplitterIP)
