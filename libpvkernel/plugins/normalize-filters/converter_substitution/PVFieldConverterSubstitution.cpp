/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldConverterSubstitution.h"

#include <fstream>

/******************************************************************************
 *
 * PVFilter::PVFieldGUIDToIP::PVFieldGUIDToIP
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
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldConverterSubstitution)
{
	PVCore::PVArgumentList args;

	args["path"] = QString();
	args["default_value"] = QString();
	args["use_default_value"] = false;
	args["sep"] = QChar(',');
	args["quote"] = QChar('"');

	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitution::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldConverterSubstitution::one_to_one(PVCore::PVField& field)
{
	auto it = _key.find(std::string(field.begin(), field.size()));

	if (it == _key.end()) {
		if (_use_default_value) {
			field.allocate_new(_default_value.size());
			field.set_end(std::copy(_default_value.begin(), _default_value.end(), field.begin()));
		}
	} else {
		std::string const& s = it->second;
		field.allocate_new(s.size());
		field.set_end(std::copy(s.begin(), s.end(), field.begin()));
	}

	return field;
}

IMPL_FILTER(PVFilter::PVFieldConverterSubstitution)
