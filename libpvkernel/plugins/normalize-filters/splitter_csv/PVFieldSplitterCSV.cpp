/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVFieldSplitterCSV.h"

PVFilter::PVFieldSplitterCSV::PVFieldSplitterCSV(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(PVFilter::PVFieldSplitterCSV, args);
}

void PVFilter::PVFieldSplitterCSV::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_sep = args.at("sep").toChar().toLatin1();
	_quote = args.at("quote").toChar().toLatin1();
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterCSV)
{
	PVCore::PVArgumentList args;
	args["sep"] = QVariant(QChar(','));
	args["quote"] = QVariant(QChar('"'));
	return args;
}

PVCore::list_fields::size_type PVFilter::PVFieldSplitterCSV::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	// FIXME : We should handle double Quote as escaped quote
	PVCore::list_fields::value_type elt(field);
	assert(elt.begin() == field.begin());
	char* cstr = elt.begin();
	size_t n = 0;
	for (size_t i = 0; i < field.size(); i++) {
		if (cstr[i] == _sep) {
			/**
			 * Don't allow the number of created fields to be greater than expected
			 * In such a case, the last field can contain one or more separators
			 */
			if (n == (_fields_expected - 1)) {
				break;
			}
			elt.set_end(cstr + i);
			elt.set_physical_end(cstr + i);
			l.insert(it_ins, elt);
			elt.set_begin(cstr + i + 1);
			n++;
		} else if (cstr[i] == _quote and (i == 0 or (i > 0 and cstr[i - 1] != '\\'))) {
			elt.set_begin(cstr + i + 1);
			do {
				i = std::find(cstr + i + 1, cstr + field.size(), _quote) - cstr;
			} while (cstr[i - 1] == '\\' and i != field.size());
			if (i == field.size()) {
				return 0;
			}
			elt.set_end(cstr + i);
			elt.set_physical_end(cstr + i);
			l.insert(it_ins, elt);
			i = std::find(cstr + i + 1, cstr + field.size(), _sep) - cstr;
			n++;
			if (i == field.size())
				return n;
			elt.set_begin(cstr + i + 1);
		}
	}
	elt.set_end(field.end());
	elt.set_physical_end(field.end());
	l.insert(it_ins, elt);
	n++;
	return n;
}

IMPL_FILTER(PVFilter::PVFieldSplitterCSV)
