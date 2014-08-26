/**
 * \file PVFieldConverterValueMapper.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVFIELDCONVERTERVALUEMAPPER_H
#define PVFILTER_PVFIELDCONVERTERVALUEMAPPER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <unordered_map>

#include <QString>

extern "C" {
// ICU
#include <unicode/ucsdet.h>
#include <unicode/ucnv.h>
}

namespace PVFilter {

namespace __impl {

struct csv_infos {

	bool first_field = true;
	std::string key;
	UChar* default_value_utf16;
	size_t default_value_len_utf16;
	typedef std::pair<UChar*, size_t> utf16_string_t;
	std::unordered_map<std::string, utf16_string_t> map;
	std::string charset = "UTF-8";
	UConverter* ucnv = nullptr;

	~csv_infos()
	{
		ucnv_close(ucnv);
	}
};

}

class PVFieldConverterValueMapper : public PVFieldsConverter {

public:
	PVFieldConverterValueMapper(PVCore::PVArgumentList const& args = PVFieldConverterValueMapper::default_args());

public:
	void init() override;
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::PVField& one_to_one(PVCore::PVField& field) override;

private:
	static void csv_new_field(void* s, size_t len, void* p);
	static void csv_new_row(int c, void* p);

private:
	QString _path;
	QString _default_value;
	bool    _use_default_value;
	QChar   _sep_char;
	QChar   _quote_char;

	__impl::csv_infos _csv_infos;
	bool _passthru = false;

	CLASS_FILTER(PVFilter::PVFieldConverterValueMapper)
};

}

#endif // PVFILTER_PVFIELDCONVERTERVALUEMAPPER_H
