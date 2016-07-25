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

	// valid indexes are in range [0;2] for IPv4 and [0;6] for IPv6
	const size_t max_valid_index_value = _ipv6 ? 6 : 2;

	const int max_params_size = _ipv6 ? 7 : 3;

	// Compute adjacente difference to have "number of elements to search" from current position.
	_indexes.clear();

	const auto param_list = params.split(sep, QString::SkipEmptyParts);

	if (param_list.size() > max_params_size) {
		throw PVFilter::PVFieldsFilterInvalidArguments(
		    (std::string("Invalid IP splitter : '") + params.toStdString() + "'").c_str());
	}

	for (const QString& s : param_list) {
		size_t i = s.toUInt();

		if (i > max_valid_index_value) {
			throw PVFilter::PVFieldsFilterInvalidArguments(
			    (std::string("Invalid IP splitter : '") + params.toStdString() + "'").c_str());
		}
		_indexes.push_back(i + 1);
	}

	// add the past-the-end index to compute full indexes set
	if (_ipv6) {
		_indexes.push_back(8);
	} else {
		_indexes.push_back(4);
	}

	std::sort(_indexes.begin(), _indexes.end());
	std::unique(_indexes.begin(), _indexes.end());
	std::adjacent_difference(_indexes.begin(), _indexes.end(), _indexes.begin());
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterIP)
{
	PVCore::PVArgumentList args;
	args["ipv6"] = false;
	args["params"] = PVFieldSplitterIP::params_ipv4;
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
