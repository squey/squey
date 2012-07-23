/**
 * \file PVFieldSplitterRegexp.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include "PVFieldSplitterRegexp.h"
#include <pvkernel/core/PVBufferSlice.h>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldSplitterRegexp, args);
}

PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp(const PVFieldSplitterRegexp& src) :
	PVFieldsFilter<PVFilter::one_to_many>(src)
{
	_regexp = src._regexp;
	_full_line = src._full_line;
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
{
	PVCore::PVArgumentList args;
	args["regexp"] = PVCore::PVArgument(QString(""));
	args["full-line"] = PVCore::PVArgument(true);
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterRegexp::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
#ifdef PROCESS_REGEXP_ICU
	QString pattern = args["regexp"].toString();
	UnicodeString icu_pat;
	icu_pat.setTo(false, (const UChar *)(pattern.constData()), pattern.size());
	UErrorCode err = U_ZERO_ERROR;
	UErrorCode pe = U_ZERO_ERROR;
	_regexp.reset(RegexPattern::compile(icu_pat, pe, err));
	if (U_FAILURE(err)) {
		PVLOG_WARN("Unable to compile pattern '%s' with ICU !\n", qPrintable(pattern));
		_valid_rx = false;
	}
	else {
		_valid_rx = true;
	}
#else
	_regexp.setPattern(args["regexp"].toString());
	_valid_rx = true;
#endif
	_full_line = args["full-line"].toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::one_to_many
 *
 *****************************************************************************/

PVCore::list_fields::size_type PVFilter::PVFieldSplitterRegexp::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	if (!_valid_rx) {
		field.set_invalid();
		return 0;
	}
#ifdef PROCESS_REGEXP_ICU
	if (_regexp_matcher_thread.get() == NULL || _regexp_pattern_thread.get() == NULL || *(_regexp_pattern_thread.get()) !=  _regexp.get()) {
		UErrorCode err = U_ZERO_ERROR;
		_regexp_pattern_thread.reset(new RegexPattern*(_regexp.get()));
		_regexp_matcher_thread.reset(new RegexMatcher*(_regexp->matcher(err)));
	}
	PVCore::list_fields::size_type n = field.split_regexp<PVCore::list_fields>(l, *(*_regexp_matcher_thread), it_ins, _full_line);
#else
	QRegExp regexp(_regexp);
	PVCore::list_fields::size_type n = field.split_regexp<PVCore::list_fields>(l, regexp, it_ins, _full_line);
#endif
	if (n == 0) {
		field.set_invalid();
	}
	return n;
}

IMPL_FILTER(PVFilter::PVFieldSplitterRegexp)
