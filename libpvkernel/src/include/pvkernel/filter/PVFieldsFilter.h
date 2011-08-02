//! \file PVFieldsFilter.h
//! $Id: PVFieldsFilter.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDSFILTER_H
#define PVFILTER_PVFIELDSFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <map>
#include <list>
#include <utility>
#include <QString>

namespace PVFilter {

enum fields_filter_type {
	one_to_one,
	one_to_many,
	many_to_many
};

typedef std::list< std::pair<PVCore::PVArgumentList, PVCore::list_fields> > list_guess_result_t;

class PVFieldsBaseFilter: public PVFilterFunction< PVCore::list_fields, PVFieldsBaseFilter >
{
public:
	typedef boost::shared_ptr<PVFieldsBaseFilter> p_type;
	typedef PVFilterFunction< PVCore::list_fields, PVFieldsBaseFilter >::func_type func_type;
public:
	PVFieldsBaseFilter() :
		PVFilterFunction< PVCore::list_fields, PVFieldsBaseFilter>()
	{
	}
};

template <fields_filter_type Ttype = many_to_many>
class PVFieldsFilter : public PVFieldsBaseFilter {
public:
	typedef PVFieldsFilter<Ttype> FilterT;
	typedef boost::shared_ptr< PVFieldsFilter<Ttype> > p_type;
	typedef PVFieldsBaseFilter base_registrable;
	typedef PVFieldsFilter<Ttype> RegAs;

public:
	PVFieldsFilter() :
		PVFieldsBaseFilter()
	{
		_type = Ttype;
		_fields_expected = 0;
	}
public:
	static fields_filter_type type() { return Ttype; };
	static QString type_name();

	// Argument guessing interface. Used by the format builder in order
	// to guess the first filter that could be applied to an input
	virtual bool guess(list_guess_result_t& /*res*/, PVCore::PVField const& /*in_field*/) { return false; } 

	// Default interface (many-to-many)
	virtual PVCore::list_fields& operator()(PVCore::list_fields &fields);

	void set_number_expected_fields(size_t n)
	{
		PVLOG_DEBUG("(PVFieldsFilter) %d (0x%x): expected %d fields\n", Ttype, this, n);
		_fields_expected = n;
	}

protected:
	// Defines field interfaces
	
	// one-to-one interface
	virtual PVCore::PVField& one_to_one(PVCore::PVField &f)
	{
		PVLOG_WARN("(PVFieldsFilter) default one_to_one function called !\n");
		return f;
	}

	// one-to-many interface
	// PVFieldsFilter is responsible for removing the original element
	// Returns the number of elements inserted
	virtual PVCore::list_fields::size_type one_to_many(PVCore::list_fields& list_ins, PVCore::list_fields::iterator it_ins, PVCore::PVField &f)
	{
		PVLOG_WARN("(PVFieldsFilter) default one_to_many function called !\n");
		list_ins.insert(it_ins, f);
		return 1;
	}

protected:
	fields_filter_type _type;
	// Defines the number of expected children. 0 means that this information is unavailable.
	size_t _fields_expected;
};

typedef PVFieldsBaseFilter::func_type PVFieldsBaseFilter_f;
typedef PVFieldsBaseFilter::p_type PVFieldsBaseFilter_p;

typedef PVFieldsBaseFilter PVFieldsFilterReg;
typedef PVFieldsFilterReg::p_type PVFieldsFilterReg_p;

typedef PVFilter::PVFieldsFilter<PVFilter::one_to_many> PVFieldsSplitter;
typedef PVFieldsSplitter::p_type PVFieldsSplitter_p;

// WARNING: all the different PVFilterLibrary's must be defined here, so that they will be exported by the DLL and imported by the others
#ifdef WIN32
LibKernelDeclExplicitTempl PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilter<PVFilter::one_to_many>::FilterT>;
LibKernelDeclExplicitTempl PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilter<PVFilter::one_to_one>::FilterT>;
LibKernelDeclExplicitTempl PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilter<PVFilter::many_to_many>::FilterT>;
LibKernelDeclExplicitTempl PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilterReg>;
LibKernelDeclExplicitTempl PVFilter::PVFieldsFilter<PVFilter::one_to_many>;
LibKernelDeclExplicitTempl PVFilter::PVFieldsFilter<PVFilter::one_to_one>;
LibKernelDeclExplicitTempl PVFilter::PVFieldsFilter<PVFilter::many_to_many>;
#endif

}

#endif
