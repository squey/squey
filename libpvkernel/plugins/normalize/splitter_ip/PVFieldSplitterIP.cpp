/**
 * \file PVFieldSplitterIP.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include "PVFieldSplitterIP.h"


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
	_ipv6 = args["ipv6"].toBool();
	_params = args["params"].toString();

	_indexes.clear();
	for (const QString& s : _params.split(sep, QString::SkipEmptyParts)) {
		_indexes.push_back(s.toUInt());
	}
	std::sort(_indexes.begin(), _indexes.end());
	std::unique(_indexes.begin(), _indexes.end());
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterIP)
{
	PVCore::PVArgumentList args;
	args["ipv6"] = false;
	args["params"] = "";
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

	if (_ipv6) {
		const size_t canonical_ipv6_max_length = 32+7;
		if (field.size() < canonical_ipv6_max_length)
		{
			for (size_t i = 0; i < _fields_expected; i++) {
				PVCore::PVField &ins_f(*l.insert(it_ins, field));
				ins_f.allocate_new(0);
				ins_f.set_invalid();
			}
			return _fields_expected;
		}

		size_t start = 0;
		for (size_t i = 0; i < _indexes.size()+1; i++) {
			size_t index = _indexes[i];
			size_t end = (i<_indexes.size()) ? ((index+1)*4+index)*2 : field.size();
			size_t length = end-start;

			PVCore::PVField &ins_f(*l.insert(it_ins, field));
			ins_f.allocate_new(length);
			memcpy(ins_f.begin(), field.begin()+start, length);
			ins_f.set_end(ins_f.begin() + length);

			start = end+2;
			ret++;
		}
	}
	else { // IPv4
		QString field_str;
		field.get_qstr(field_str);

		QRegExp ipv4_regex;
		ipv4_regex.setPattern("^\\d+\\.\\d+\\.\\d+\\.\\d+$");
		if (!ipv4_regex.exactMatch(field_str))
		{
			for (size_t i = 0; i < _fields_expected; i++) {
				PVCore::PVField &ins_f(*l.insert(it_ins, field));
				ins_f.allocate_new(0);
				ins_f.set_invalid();
			}
			return _fields_expected;
		}

		int last_index = -1;
		size_t start = 0;
		size_t end;
		size_t length;
		for (size_t index : _indexes) {
			size_t pos = start/2;
			for (size_t j = 0; j < (index-last_index); j++) {
				pos = field_str.indexOf(".", pos+1);
			}
			end = pos*2;
			length = end-start;

			PVCore::PVField &ins_f(*l.insert(it_ins, field));
			ins_f.allocate_new(length);
			memcpy(ins_f.begin(), field.begin()+start, length);
			ins_f.set_end(ins_f.begin() + length);
			ret++;

			start = end+2;
			last_index = index;
		}

		length = field.size()-start;
		PVCore::PVField &ins_f(*l.insert(it_ins, field));
		ins_f.allocate_new(length);
		memcpy(ins_f.begin(), field.begin()+start, length);
		ins_f.set_end(ins_f.begin() + length);
		ret++;
	}

	return ret;
}


IMPL_FILTER(PVFilter::PVFieldSplitterIP)
