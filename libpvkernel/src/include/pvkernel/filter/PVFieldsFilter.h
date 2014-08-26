/**
 * \file PVFieldsFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVFIELDSFILTER_H
#define PVFILTER_PVFIELDSFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <map>
#include <list>
#include <vector>
#include <utility>
#include <QString>
#include <QHash>

namespace PVFilter {

enum fields_filter_type {
	one_to_one,
	one_to_many,
	many_to_many
};

typedef std::list< std::pair<PVCore::PVArgumentList, PVCore::list_fields> > list_guess_result_t;
// Associate the tags to their columns
typedef QHash<QString, PVCol> filter_child_axes_tag_t;

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

	virtual void init() {}

	virtual void set_children_axes_tag(filter_child_axes_tag_t const& axes)
	{
		filter_child_axes_tag_t::const_iterator it;
		for (it = axes.begin(); it != axes.end(); it++) {
			PVLOG_DEBUG("(PVFieldsFilter) axis tag %s set for col %d.\n", qPrintable(it.key()), it.value());
		}
		_axes_tag = axes;
	}

protected:
	bool is_tag_present(QString const& tag)
	{
		return _axes_tag.contains(tag);
	}

	PVCol get_col_for_tag(QString const& tag)
	{
		assert(_axes_tag.contains(tag));
		return _axes_tag[tag];
	}

protected:
	filter_child_axes_tag_t _axes_tag;
};

template <fields_filter_type Ttype = many_to_many>
class PVFieldsFilter : public PVFieldsBaseFilter {
public:
	typedef PVFieldsFilter<Ttype> FilterT;
	//typedef PVFieldsBaseFilter base_registrable;
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

	// Filter interface (many-to-many)
	PVCore::list_fields& operator()(PVCore::list_fields &fields);

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

	// many-to-many interface
	virtual PVCore::list_fields& many_to_many(PVCore::list_fields& fields)
	{
		PVLOG_WARN("(PVFieldsFilter) default many_to_many function called !\n");
		return fields;
	}

protected:
	fields_filter_type _type;
	// Defines the number of expected children. 0 means that this information is unavailable.
	size_t _fields_expected;

	CLASS_FILTER_NONREG_NOPARAM(FilterT)

	// Custom registration functions (CLASS_REGISTRABLE should be used here, but we have issues under Windows)
public:
	typedef boost::shared_ptr<FilterT> p_type;
protected:
	virtual base_registrable* _clone_me() const { FilterT* ret = new FilterT(*this); return ret; }
};

// Macro for reporting invalid fields
#define PVLOG_WARN_FIELD(field, str, ...) PVLOG_WARN(str " ; in index (1-based) %d in source %s \n", __VA_ARGS__, field.get_index_of_parent_element()+1, qPrintable(field.elt_parent()->chunk_parent()->source()->human_name()))

typedef PVFieldsBaseFilter::func_type PVFieldsBaseFilter_f;
typedef PVFieldsBaseFilter::p_type PVFieldsBaseFilter_p;

typedef PVFieldsBaseFilter PVFieldsFilterReg;
typedef PVFieldsFilterReg::p_type PVFieldsFilterReg_p;

typedef PVFilter::PVFieldsFilter<PVFilter::one_to_many> PVFieldsSplitter;
typedef PVFieldsSplitter::p_type PVFieldsSplitter_p;

typedef PVFilter::PVFieldsFilter<PVFilter::one_to_one> PVFieldsConverter;
typedef PVFieldsConverter::p_type PVFieldsConverter_p;

// WARNING: all the different PVFilterLibrary's must be defined here, so that they will be exported by the DLL and imported by the others
#ifdef WIN32
LibKernelDeclExplicitTempl PVCore::PVClassLibrary<PVFilter::PVFieldsFilter<PVFilter::one_to_many>::FilterT>;
LibKernelDeclExplicitTempl PVCore::PVClassLibrary<PVFilter::PVFieldsFilter<PVFilter::one_to_one>::FilterT>;
LibKernelDeclExplicitTempl PVCore::PVClassLibrary<PVFilter::PVFieldsFilter<PVFilter::many_to_many>::FilterT>;
LibKernelDeclExplicitTempl PVCore::PVClassLibrary<PVFilter::PVFieldsFilterReg>;
LibKernelDeclExplicitTempl PVFilter::PVFieldsFilter<PVFilter::one_to_many>;
LibKernelDeclExplicitTempl PVFilter::PVFieldsFilter<PVFilter::one_to_one>;
LibKernelDeclExplicitTempl PVFilter::PVFieldsFilter<PVFilter::many_to_many>;
LibKernelDeclExplicitTempl PVCore::PVTag<PVFieldsFilter<PVFilter::one_to_many> >;
LibKernelDeclExplicitTempl PVCore::PVTag<PVFieldsFilter<PVFilter::one_to_one> >;
LibKernelDeclExplicitTempl PVCore::PVTag<PVFieldsFilter<PVFilter::many_to_many> >;
#endif

typedef PVCore::PVClassLibrary<PVFieldsSplitter>::tag PVFieldsSplitterTag;
typedef PVCore::PVClassLibrary<PVFieldsSplitter>::list_tags PVFieldsSplitterListTags;

}

#endif
