//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVFieldConverterSubstitution.h"

#include <pvkernel/core/PVUtils.h>

#include <fstream>

#include <boost/utility/string_ref.hpp>

/******************************************************************************
 *
 * PVFilter::ConverterSubstitution::ConverterSubstitution
 *
 *****************************************************************************/
PVFilter::PVFieldConverterSubstitution::PVFieldConverterSubstitution(
    PVCore::PVArgumentList const& args)
    : PVFieldsConverter()
{
	INIT_FILTER(PVFilter::PVFieldConverterSubstitution, args);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitution::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitution::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);

	_modes = args.at("modes").toUInt();
	_invert_order = args.at("invert_order").toBool();

	// whole field mode
	_default_value = args.at("default_value").toString().toStdString();
	_use_default_value = args.at("use_default_value").toBool();
	_sep_char = args.at("sep").toChar().toLatin1();
	_quote_char = args.at("quote").toChar().toLatin1();

	std::ifstream ifs(args.at("path").toString().toStdString());
	std::string buffer(4096 * 2, 0);
	// FIXME : Add more check on file format.
	// FIXME : Handle quote char
	while (ifs.getline(&buffer.front(), buffer.size())) {
		char* txt = &buffer.front();
		char* key = std::find(txt, txt + buffer.size(), _sep_char);
		char* v = std::find(key + 1, txt + buffer.size(), '\0');
		_key[std::string(txt, key - txt)] = std::string(key + 1, v - key - 1);
	}

	// substrings mode
	const QStringList& substrings_map =
	    PVCore::deserialize_base64<QStringList>(args.at("substrings_map").toString());
	for (int i = 0; i < ((substrings_map.size() / 2) * 2) / 2; i++) {
		_substrings_map.emplace_back(substrings_map[i * 2].toStdString(),
		                             substrings_map[i * 2 + 1].toStdString());
	}
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldConverterSubstitution)
{
	PVCore::PVArgumentList args;

	args["modes"] = 0;
	args["invert_order"] = false;

	args["path"] = QString();
	args["default_value"] = QString();
	args["use_default_value"] = false;
	args["sep"] = QChar(',');
	args["quote"] = QChar('"');

	args["substrings_map"] = QString();

	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitution::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldConverterSubstitution::one_to_one(PVCore::PVField& field)
{

	auto whole_field_f = [&]() {
		if (_modes & ESubstitutionModes::WHOLE_FIELD) {

			auto it = _key.find(std::string(field.begin(), field.size()));

			if (it == _key.end()) {
				if (_use_default_value) {
					field.allocate_new(_default_value.size());
					field.set_end(
					    std::copy(_default_value.begin(), _default_value.end(), field.begin()));
				}
			} else {
				std::string const& s = it->second;

				field.allocate_new(s.size());
				field.set_end(std::copy(s.begin(), s.end(), field.begin()));
			}
		}
	};

	auto substrings_f = [&]() {
		if (_modes & ESubstitutionModes::SUBSTRINGS) {
			for (const std::pair<std::string, std::string>& sub : _substrings_map) {
				boost::string_ref field_ref(field.begin(), field.size());
				const std::string& from = sub.first;
				const std::string& to = sub.second;

				size_t pos = field_ref.find(from);
				if (pos != std::string::npos) {
					std::string field_copy = field_ref.to_string();
					PVCore::replace(field_copy, from, to, pos);

					field.allocate_new(field_copy.size());
					field.set_end(std::copy(field_copy.begin(), field_copy.end(), field.begin()));
				}
			}
		}
	};

	_invert_order ? substrings_f(), whole_field_f() : whole_field_f(), substrings_f();

	return field;
}
