/**
 * \file PVFieldConverterSubstitution.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldConverterSubstitution.h"

#include <QFile>
#include <QTextStream>

#include <pvkernel/widgets/qkeysequencewidget.h>
#include <pvkernel/rush/PVCharsetDetect.h>
#include <pvkernel/core/PVUtils.h>

extern "C" {
// libcsv
#include "libcsv.h"
}

/******************************************************************************
 *
 * PVFilter::PVFieldGUIDToIP::PVFieldGUIDToIP
 *
 *****************************************************************************/
PVFilter::PVFieldConverterSubstitution::PVFieldConverterSubstitution(PVCore::PVArgumentList const& args) :
	PVFieldsConverter()
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

	_path              = args["path"].toString();
	_default_value     = args["default_value"].toString();
	_use_default_value = args["use_default_value"].toBool();
	_sep_char          = args["sep"].toChar();
	_quote_char        = args["quote"].toChar();
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitution::init
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitution::init()
{
	// Initialize ICU
	UErrorCode status = U_ZERO_ERROR;
	_csv_infos.ucnv = ucnv_open(_csv_infos.charset.c_str(), &status);

	// Parse using libCSV
	csv_parser p;
	if (csv_init(&p, 0) != 0) {
		PVLOG_ERROR("Unable to initialize libcsv !\n");
	}
	csv_set_delim(&p, _sep_char.toAscii());
	csv_set_quote(&p, _quote_char.toAscii());
	QFile f(_path);
	if (!f.open(QFile::ReadOnly | QFile::Text)) {
		PVLOG_WARN("Filter '%s' of type '%s' was unable to open conversion file !\n", qPrintable(type_name()), qPrintable(registered_name()));
		_passthru = true;
		return;
	}
	QTextStream in(&f);
	QString content = in.readAll();
	PVRush::PVCharsetDetect cd;
	if (cd.HandleData(content.toStdString().c_str(), content.length()) == NS_OK) {
		cd.DataEnd();
		if (cd.found()) {
			_csv_infos.charset = cd.GetCharset();
		}
	}
	csv_parse(&p, content.toLocal8Bit().data(), content.size(), &PVFilter::PVFieldConverterSubstitution::csv_new_field, &PVFilter::PVFieldConverterSubstitution::csv_new_row, (void*)&_csv_infos);
	csv_fini(&p, &PVFilter::PVFieldConverterSubstitution::csv_new_field, &PVFilter::PVFieldConverterSubstitution::csv_new_row, (void*)&_csv_infos);
	csv_free(&p);
	if (!_csv_infos.map.size()) {
		PVLOG_WARN("Filter '%s' of type '%s' was unable to detect any value mapping !\n", qPrintable(type_name()), qPrintable(registered_name()));
		_passthru = true;
		return;
	}

	// Convert default value to UTF-16
	if (_use_default_value) {
		status = U_ZERO_ERROR;
		static tbb::tbb_allocator<UChar> alloc;
		_csv_infos.default_value_len_utf16 = _default_value.length() * 2;
		_csv_infos.default_value_utf16 = alloc.allocate(_csv_infos.default_value_len_utf16);
		UChar* target = _csv_infos.default_value_utf16;
		const UChar* target_end = target + _csv_infos.default_value_len_utf16;
		const char* data_conv = _default_value.toStdString().c_str();
		const char* data_conv_end = _default_value.toStdString().c_str() + _default_value.length();

		ucnv_toUnicode(_csv_infos.ucnv, &target, target_end, &data_conv, data_conv_end, NULL, true, &status);
	}
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldConverterSubstitution)
{
	PVCore::PVArgumentList args;

	args["path"]              = QString();
	args["default_value"]     = QString();
	args["use_default_value"] = false;
	args["sep"]               = QChar(',');
	args["quote"]             = QChar('"');

	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitution::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldConverterSubstitution::one_to_one(PVCore::PVField& field)
{
	if (!unlikely(_passthru)) {
		QString str_tmp;
		std::string f = field.get_qstr(str_tmp).toStdString();

		__impl::csv_infos::utf16_map_t::const_iterator iter = _csv_infos.map.find(f); // not using at() because throwing exceptions really really kills the perfs

		if(iter != _csv_infos.map.end()) { // found
			const __impl::csv_infos::utf16_string_t& mapped_field = iter->second;
			size_t mapped_field_len_utf16 = mapped_field.second;
			field.allocate_new(mapped_field_len_utf16);
			memcpy(field.begin(), mapped_field.first, mapped_field_len_utf16);
			field.set_end(field.begin() + mapped_field_len_utf16);
		}
		else { // not found
			if (_use_default_value) {
				field.allocate_new(_csv_infos.default_value_len_utf16);
				memcpy(field.begin(), _csv_infos.default_value_utf16, _csv_infos.default_value_len_utf16);
				field.set_end(field.begin() + _csv_infos.default_value_len_utf16);
			}
		}
	}

	return field;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitution::csv_new_field
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitution::csv_new_field(void* s, size_t len, void* p)
{
	__impl::csv_infos* infos = (__impl::csv_infos*) p;

	std::string str = std::string((const char*)s, len);

	if (infos->first_field) {
		infos->key = str;
	}
	else {
		static tbb::tbb_allocator<UChar> alloc;

		size_t mapped_field_len_utf16_max = len * 2;
		UChar* output = alloc.allocate(mapped_field_len_utf16_max);
		UChar* target = output;
		const UChar* target_end = target + mapped_field_len_utf16_max;
		const char* data_conv = str.c_str();
		const char* data_conv_end = str.c_str() + str.length();

		UErrorCode status = U_ZERO_ERROR;
		ucnv_toUnicode(infos->ucnv, &target, target_end, &data_conv, data_conv_end, NULL, true, &status);

		const size_t mapped_field_len_utf16 = (uintptr_t)target - (uintptr_t)output;
		infos->map[infos->key] = __impl::csv_infos::utf16_string_t(output, mapped_field_len_utf16);
	}

	infos->first_field = false;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitution::csv_new_row
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitution::csv_new_row(int /*c*/, void* p)
{
	__impl::csv_infos* infos = (__impl::csv_infos*) p;

	infos->first_field = true;
}


IMPL_FILTER(PVFilter::PVFieldConverterSubstitution)
