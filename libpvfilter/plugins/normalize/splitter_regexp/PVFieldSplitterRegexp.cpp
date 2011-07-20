//! \file PVCore::PVFieldSplitterRegexp.cpp
//! $Id: PVFieldSplitterRegexp.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include "PVFieldSplitterRegexp.h"
#include <pvcore/PVBufferSlice.h>

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
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
{
	PVCore::PVArgumentList args;
	args["regexp"] = PVCore::PVArgument(QString("^(.*)$"));
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
	_regexp_matcher.reset(_regexp->matcher(err));
	if (U_FAILURE(err)) {
		PVLOG_WARN("Unable to compile pattern '%s' with ICU !\n", qPrintable(pattern));
	}
#else
	_regexp.setPattern(args["regexp"].toString());
#endif
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::one_to_many
 *
 *****************************************************************************/

PVCore::list_fields::size_type PVFilter::PVFieldSplitterRegexp::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
/*	if (_regexp_thread.get() == NULL) {
#ifdef PROCESS_REGEXP_ICU
		UErrorCode err = U_ZERO_ERROR;
		_regexp_thread.reset(_regexp->matcher(err));
#else
		_regexp_thread.reset(new QRegExp(_regexp));
#endif
	}*/
#ifdef PROCESS_REGEXP_ICU
	// AG: Disable ICU for now...
	// See ticket 1
#else
	QRegExp regexp(_regexp);
#endif
	PVCore::list_fields::size_type n = field.split_regexp<PVCore::list_fields>(l, regexp, it_ins);
	if (n == 0) {
		field.set_invalid();
	}
	return n;
}

IMPL_FILTER(PVFilter::PVFieldSplitterRegexp)
