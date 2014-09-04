/**
 * \file PVFieldSplitterKeyValue.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldSplitterKeyValue.h"

#include <stdio.h>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::PVFieldSplitterKeyValue
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterKeyValue::PVFieldSplitterKeyValue(PVCore::PVArgumentList const& args) :
	PVFieldsSplitter()
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

	_separator = args["sep"].toString();
	_quote     = args["quote"].toChar();
	_affect    = args["affectation"].toString();
	_keys 	   = args["keys"].toStringList();
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterKeyValue)
{
	PVCore::PVArgumentList args;

	args["sep"]         = QString("; ");
	args["quote"]       = QChar('"');
	args["affectation"] = QString(": ");
	args["keys"]        = QStringList();

	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::init
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterKeyValue::init()
{
	// Initialize ICU
	UErrorCode status = U_ZERO_ERROR;
	_ucnv = ucnv_open("UTF-8", &status);

	auto convert_to_utf16 = [&](const char* str, size_t len) -> std::string
	{
		static tbb::tbb_allocator<UChar> alloc;

		size_t str_len_utf16_max = len * 2;
		UChar* output = alloc.allocate(str_len_utf16_max);
		UChar* target = output;
		const UChar* target_end = target + str_len_utf16_max;
		const char* data_conv = str;
		const char* data_conv_end = str + len;

		UErrorCode status = U_ZERO_ERROR;
		ucnv_toUnicode(_ucnv, &target, target_end, &data_conv, data_conv_end, NULL, true, &status);
		const size_t str_len_utf16 = (uintptr_t)target - (uintptr_t)output;

		return std::string((const char*)output, str_len_utf16);
	};

	// Convert keys and parameters to UTF-16
	size_t i = 0;
	for (const QString& key: _keys) {
		std::string key_utf16 = convert_to_utf16(key.toStdString().c_str(), key.size());
		_keys_map[key_utf16] = i++;
	}
	_separator_utf16 = convert_to_utf16(_separator.toStdString().c_str(), _separator.size());
	_affect_utf16 = convert_to_utf16(_affect.toStdString().c_str(),  _affect.size());
	_quote_utf16 = convert_to_utf16(QString(_quote).toStdString().c_str(), 2);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::~PVFieldSplitterKeyValue
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterKeyValue::~PVFieldSplitterKeyValue()
{
	ucnv_close(_ucnv);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterKeyValue::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	PVCore::list_fields::size_type ret = 0;

	const char* sep = _separator_utf16.c_str();
	size_t sep_len = _separator_utf16.size();

	const char* affect = _affect_utf16.c_str();
	size_t affect_len = _affect_utf16.size();

	const char* quote = _quote_utf16.c_str();
	size_t quote_len = _quote_utf16.size();

	size_t childen_count = _keys.size();

	struct utf16_str_t {
		utf16_str_t() : buffer(nullptr), length(0) {}
		utf16_str_t(const char* b, size_t l) : buffer(b), length(l) {}
		const char* buffer;
		size_t length;
	};

	std::vector<utf16_str_t> values;
	values.reserve(childen_count);
	for (size_t i = 0; i < childen_count; i++) {
		values.push_back(utf16_str_t());
	}

	int key_position = -1;
	tokenize(field.begin(), field.size(), sep, sep_len, quote, quote_len, [&](const char* token, size_t token_size)
	{
		tokenize(token, token_size, affect, affect_len, quote, quote_len, [&](const char* keyvalue, size_t keyvalue_size)
		{
			if (key_position == -1) { // key
				keys_map_t::const_iterator iter = _keys_map.find(std::move(std::string(keyvalue, keyvalue_size)));

				if(iter != _keys_map.end()) {
					key_position = iter->second;
				}
			}
			else { // value
				values[key_position] = utf16_str_t(keyvalue, keyvalue_size);

				key_position = -1;
			}
		});
	});

	for (size_t i = 0; i < childen_count; i++) {
		const utf16_str_t& value = values[i];

		PVCore::PVField &ins_f(*l.insert(it_ins, field));
		ins_f.allocate_new(value.length);
		memcpy(ins_f.begin(), value.buffer, value.length);
		ins_f.set_end(ins_f.begin() + value.length);
	}

	return childen_count;
}
