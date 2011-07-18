//! \file PVCore::PVFieldSplitterRegexp.h
//! $Id: PVFieldSplitterRegexp.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDSPLITTERREGEXP_H
#define PVFILTER_PVFIELDSPLITTERREGEXP_H

//#define PROCESS_REGEXP_ICU

#include <pvcore/general.h>
#include <pvcore/PVField.h>
#include <pvfilter/PVFieldsFilter.h>
#include <boost/thread/tss.hpp>
#ifdef PROCESS_REGEXP_ICU
#include <boost/shared_ptr.hpp>
#include <unicode/regex.h>
#else
#include <QRegExp>
#endif


namespace PVFilter {

class PVFieldSplitterRegexp : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterRegexp(PVArgumentList const& args = PVFieldSplitterRegexp::default_args());
	PVFieldSplitterRegexp(const PVFieldSplitterRegexp& src);
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);
public:
	virtual void set_args(PVArgumentList const& args);
protected:
#ifdef PROCESS_REGEXP_ICU
	boost::shared_ptr<RegexPattern> _regexp;
	boost::shared_ptr<RegexMatcher> _regexp_matcher;
	boost::thread_specific_ptr<RegexMatcher> _regexp_thread;
#else
	QRegExp _regexp;
	//boost::thread_specific_ptr<QRegExp> _regexp_thread;
#endif

	CLASS_FILTER(PVFilter::PVFieldSplitterRegexp)
};

}

#endif
