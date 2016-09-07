/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVFILTER_PVFIELDSPLITTERLENGTH_H
#define PVFILTER_PVFIELDSPLITTERLENGTH_H

#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

class PVFieldSplitterLength : public PVFieldsFilter<one_to_many>
{
  public:
	static const constexpr char* param_length = "length";
	static const constexpr char* param_from_left = "from_left";

  public:
	PVFieldSplitterLength(PVCore::PVArgumentList const& args = default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;

  public:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field) override;

  protected:
	CLASS_FILTER(PVFilter::PVFieldSplitterLength)

  private:
	size_t _length;
	bool _from_left;
};
}

#endif // PVFILTER_PVFIELDSPLITTERLENGTH_H
