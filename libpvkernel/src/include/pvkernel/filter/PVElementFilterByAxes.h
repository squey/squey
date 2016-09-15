/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#ifndef PVFILTER_PVELEMENTFILTERBYAXES_H
#define PVFILTER_PVELEMENTFILTERBYAXES_H

#include <pvkernel/rush/PVFormat.h> // for PVFormat, etc

#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <pvkernel/core/PVElement.h> // for PVElement

namespace PVFilter
{

class PVElementFilterByAxes : public PVElementFilterByFields
{
  public:
	using fields_mask_t = PVRush::PVFormat::fields_mask_t;

  public:
	PVElementFilterByAxes(const fields_mask_t& fields_mask);

  public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt);

  protected:
	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterByAxes)

  private:
	const fields_mask_t& _fields_mask;
};

} // namespace PVFilter

#endif // PVFILTER_PVELEMENTFILTERBYAXES_H
