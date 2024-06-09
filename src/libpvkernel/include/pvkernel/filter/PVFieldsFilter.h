/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVFILTER_PVFIELDSFILTER_H
#define PVFILTER_PVFIELDSFILTER_H

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <qcontainerfwd.h>
#include <qlist.h>
#include <stddef.h>
#include <list>
#include <QString>
#include <QHash>
#include <memory>
#include <stdexcept>
#include <utility>

#include "pvkernel/core/PVArgument.h"
#include "pvkernel/core/PVLogger.h"

namespace PVCore {
class PVField;
}  // namespace PVCore

namespace PVFilter
{

class PVFieldsFilterInvalidArguments : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};

enum fields_filter_type { one_to_one, one_to_many, many_to_many };

typedef std::list<std::pair<PVCore::PVArgumentList, PVCore::list_fields>> list_guess_result_t;

class PVFieldsBaseFilter : public PVFilterFunction<PVCore::list_fields, PVFieldsBaseFilter>
{
  public:
	typedef std::shared_ptr<PVFieldsBaseFilter> p_type;
	typedef PVFilterFunction<PVCore::list_fields, PVFieldsBaseFilter>::func_type func_type;

  public:
	PVFieldsBaseFilter() : PVFilterFunction<PVCore::list_fields, PVFieldsBaseFilter>() {}
};

template <fields_filter_type Ttype = many_to_many>
class PVFieldsFilter : public PVFieldsBaseFilter
{
  public:
	typedef PVFieldsFilter<Ttype> FilterT;
	// typedef PVFieldsBaseFilter base_registrable;
	typedef PVFieldsFilter<Ttype> RegAs;

  public:
	PVFieldsFilter() : PVFieldsBaseFilter() { _fields_expected = 0; }

  public:
	static fields_filter_type type() { return Ttype; };
	static QString type_name();

	// Argument guessing interface. Used by the format builder in order
	// to guess the first filter that could be applied to an input
	virtual bool guess(list_guess_result_t& /*res*/, PVCore::PVField& /*in_field*/)
	{
		return false;
	}

	// Filter interface (many-to-many)
	PVCore::list_fields& operator()(PVCore::list_fields& fields) override;

	void set_number_expected_fields(size_t n)
	{
		PVLOG_DEBUG("(PVFieldsFilter) %d (0x%x): expected %d fields\n", Ttype, this, n);
		_fields_expected = n;
	}

	void set_children_names(const QStringList& names) {
		_fields_names = names;
	}

  protected:
	// Defines field interfaces

	// one-to-one interface
	virtual PVCore::PVField& one_to_one(PVCore::PVField& f)
	{
		PVLOG_WARN("(PVFieldsFilter) default one_to_one function called !\n");
		return f;
	}

	// one-to-many interface
	// PVFieldsFilter is responsible for removing the original element
	// Returns the number of elements inserted
	virtual PVCore::list_fields::size_type one_to_many(PVCore::list_fields& list_ins,
	                                                   PVCore::list_fields::iterator it_ins,
	                                                   PVCore::PVField& f)
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
	// Defines the number of expected children. 0 means that this information is unavailable.
	size_t _fields_expected;
	QStringList _fields_names;

	CLASS_FILTER_NONREG_NOPARAM(FilterT)

	// Custom registration functions (CLASS_REGISTRABLE should be used here, but we have issues
	// under Windows)
  public:
	typedef std::shared_ptr<FilterT> p_type;

  protected:
	base_registrable* _clone_me() const override
	{
		auto ret = new FilterT(*this);
		return ret;
	}
};

typedef PVFieldsBaseFilter::p_type PVFieldsBaseFilter_p;

typedef PVFieldsBaseFilter PVFieldsFilterReg;
typedef PVFieldsFilterReg::p_type PVFieldsFilterReg_p;

typedef PVFilter::PVFieldsFilter<PVFilter::one_to_many> PVFieldsSplitter;
typedef PVFieldsSplitter::p_type PVFieldsSplitter_p;

typedef PVFilter::PVFieldsFilter<PVFilter::one_to_one> PVFieldsConverter;
typedef PVFieldsConverter::p_type PVFieldsConverter_p;
} // namespace PVFilter

#endif
