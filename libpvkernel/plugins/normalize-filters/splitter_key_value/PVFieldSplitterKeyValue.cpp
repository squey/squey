/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldSplitterKeyValue.h"
#include <unordered_map>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::PVFieldSplitterKeyValue
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterKeyValue::PVFieldSplitterKeyValue(PVCore::PVArgumentList const& args)
    : PVFieldsSplitter()
{
	INIT_FILTER(PVFilter::PVFieldSplitterKeyValue, args);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterKeyValue::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);

	_separator = args.at("sep").toString().toStdString();
	_quote = args.at("quote").toChar().toLatin1();
	_affect = args.at("affectation").toString().toStdString();
	_keys.clear();
	for (auto const& key : args.at("keys").toStringList()) {
		size_t quote_offset = key[0] == _quote and key[key.size() - 1] == _quote;
		_keys.push_back(
		    std::string(key.toStdString().c_str() + quote_offset, key.size() - (quote_offset * 2)));
	}
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterKeyValue)
{
	PVCore::PVArgumentList args;

	args["sep"] = QString("; ");
	args["quote"] = QChar('"');
	args["affectation"] = QString(": ");
	args["keys"] = QStringList();

	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterKeyValue::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	std::unordered_map<std::string, std::tuple<char*, char*>> key_view;
	char* txt = field.begin();
	while (txt < field.end()) {
		char* aff = std::find_if(txt, field.end() - _affect.size() + 1, [this](char& c) {
			return std::equal(&c, &c + _affect.size(), _affect.c_str());
		});

		char* start_value = aff + _affect.size();
		char* end = nullptr;
		char* new_txt = nullptr;
		// FIXME : We should check for escaped quote.
		if (start_value[0] == _quote) {
			start_value++;
			end = std::find(start_value, field.end(), _quote);
			new_txt = end + _separator.size() + 1;
		} else {
			end = std::find_if(start_value, field.end() - _separator.size() + 1, [this](char& c) {
				return std::equal(&c, &c + _separator.size(), _separator.c_str());
			});
			new_txt = end + _separator.size();
		}

		const char* key = txt;
		size_t key_size = aff - txt;
		size_t quote_offset = key[0] == _quote and key[key_size - 1] == _quote;
		key_view[std::string(key + quote_offset, key_size - (quote_offset * 2))] =
		    std::make_tuple(start_value, end);
		txt = new_txt;
	}

	for (auto const& key : _keys) {
		PVCore::PVField& ins_f(*l.insert(it_ins, field));
		auto v = key_view.find(key);
		if (v == key_view.end()) {
			ins_f.set_end(ins_f.begin());
		} else {
			auto const& pos = v->second;
			ins_f.set_begin(std::get<0>(pos));
			ins_f.set_end(std::get<1>(pos));
		}
	}

	return _keys.size();
}
