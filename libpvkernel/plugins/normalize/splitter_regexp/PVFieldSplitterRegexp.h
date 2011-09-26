//! \file PVCore::PVFieldSplitterRegexp.h
//! $Id: PVFieldSplitterRegexp.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDSPLITTERREGEXP_H
#define PVFILTER_PVFIELDSPLITTERREGEXP_H

#define PROCESS_REGEXP_ICU

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>
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
	PVFieldSplitterRegexp(PVCore::PVArgumentList const& args = PVFieldSplitterRegexp::default_args());
	PVFieldSplitterRegexp(const PVFieldSplitterRegexp& src);
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);
public:
	virtual void set_args(PVCore::PVArgumentList const& args);
protected:
#ifdef PROCESS_REGEXP_ICU
	boost::shared_ptr<RegexPattern> _regexp;
	// We store a pointer to a pointer because, if we only store a pointer to RegexMatcher or RegexPattern,
	// when boost::thread_specific_ptr will call the destructor function that we would have given us
	// (that does nothing), it will do so after the unloading of the shared libraries, and our destruction
	// fonction won't be available. That's a dirty solution for a silly problem, we should be able to tell
	// boost::thread_specific_ptr that we don't need any deallocation !
	boost::thread_specific_ptr<RegexMatcher*> _regexp_matcher_thread;
	boost::thread_specific_ptr<RegexPattern*> _regexp_pattern_thread;
#else
	QRegExp _regexp;
	//boost::thread_specific_ptr<QRegExp> _regexp_thread;
#endif

	bool _valid_rx;
	bool _full_line;

	CLASS_FILTER(PVFilter::PVFieldSplitterRegexp)
};

}

#endif
