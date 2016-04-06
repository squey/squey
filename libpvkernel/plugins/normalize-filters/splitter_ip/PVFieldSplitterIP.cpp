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
PVFilter::PVFieldSplitterIP::PVFieldSplitterIP(PVCore::PVArgumentList const& args) :
	PVFieldsSplitter()
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
	args["params"] = "0,1,2,3";
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterIP::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterIP::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	PVCore::list_fields::size_type ret = 0;
	char token = '.';

	if (_ipv6) {
		// WARNING: We don't check second ':' for ipv6. Is it important?
		token = ':';

		// Set empty fields if the ipv6 is not "full" and mark field as invalid.
		const size_t canonical_ipv6_max_length = 32+7;
		// FIXME : Should we pick first elements?
		std::cout << std::string(field.begin(), field.end()) << " - " << field.size() << "/" << canonical_ipv6_max_length << std::endl;
		if (field.size() < canonical_ipv6_max_length)
		{
			for (size_t i = 0; i < _fields_expected; i++) {
				PVCore::PVField &ins_f(*l.insert(it_ins, field));
				ins_f.set_end(ins_f.begin());
				ins_f.set_invalid();
			}
			return _fields_expected;
		}
	}

	char* pos = field.begin();
	PVCore::PVField f(field);
	for (size_t index : _indexes) {
		for (size_t j = 0; j < index; j++) {
			f.set_begin(pos);
			// + 1 to be after the token
			pos = std::find(pos, field.end(), token) + 1;
		}
		f.set_end(pos - 1);
		l.insert(it_ins, f);
	}

	return _indexes.size();
}


IMPL_FILTER(PVFilter::PVFieldSplitterIP)
