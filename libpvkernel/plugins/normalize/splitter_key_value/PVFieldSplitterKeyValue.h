/**
 * \file PVFieldSplitterKeyValue.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVFIELDSPLITTERKEYVALUE_H
#define PVFILTER_PVFIELDSPLITTERKEYVALUE_H

#include <unordered_map>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <QChar>
#include <QString>

extern "C" {
// ICU
#include <unicode/ucsdet.h>
#include <unicode/ucnv.h>
}

namespace PVFilter {

class PVFieldSplitterKeyValue : public PVFieldsSplitter {

public:
	PVFieldSplitterKeyValue(PVCore::PVArgumentList const& args = PVFieldSplitterKeyValue::default_args());
	~PVFieldSplitterKeyValue();

public:
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field) override;

protected:
	void init() override;

private:
	template <typename F>
	void tokenize(
		const char* str,
		size_t str_len,
		const char* sep,
		size_t sep_len,
		const char* quote,
		size_t quote_len,
		const F& f
	) const
	{
		size_t pos          = 0;
		size_t last_pos     = 0;
		size_t quote_pos[2] = {0, 0};
		bool quote_on       = false;

		auto call_func = [&]()
		{
			// Remove potential enclosing quotes before calling the provided function
			size_t offset = 0;
			if (quote_pos[0] == last_pos && quote_pos[1] == (pos -2)) {
				offset = 2;
			}
			f(str + last_pos + offset, pos - last_pos - (2*offset));
		};

		for (pos = 0; pos < str_len; pos += 2) {
			if (strncmp(&str[pos], sep, sep_len) == 0 && !quote_on) {
				call_func();
				last_pos = pos + sep_len;
				quote_on = false;
				quote_pos[0] = quote_pos[1] = 0;
			}
			else if (strncmp(&str[pos], quote, quote_len) == 0) {
				quote_pos[quote_on] = pos;
				quote_on = !quote_on;
			}
		}
		call_func();
	}

private:
	QString     _separator;
	QChar       _quote;
	QString     _affect;

	std::string _separator_utf16;
	std::string _quote_utf16;
	std::string _affect_utf16;

	QStringList _keys;

	typedef std::unordered_map<std::string, size_t> keys_map_t;
	keys_map_t _keys_map;

	UConverter* _ucnv = nullptr;

	CLASS_FILTER(PVFilter::PVFieldSplitterKeyValue)
};

}

#endif // PVFILTER_PVFIELDSPLITTERKEYVALUE_H
