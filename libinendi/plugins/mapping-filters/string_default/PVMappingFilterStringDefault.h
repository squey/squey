/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * Compute string default mapping.
 *
 * This mapping split first on string len, then sort them on value.
 */
class PVMappingFilterStringDefault : public PVMappingFilter
{
  public:
	PVMappingFilterStringDefault(
	    PVCore::PVArgumentList const& args = PVMappingFilterStringDefault::default_args());

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

  public:
	/**
	 * Setter for case_sensitif information.
	 */
	void set_args(PVCore::PVArgumentList const& args) override;

	/**
	 * MetaInformation for this plugin.
	 */
	QString get_human_name() const override { return "Default"; }

  private:
	bool _case_sensitive; //!< Whether we should care about case for mapping.

	CLASS_FILTER(PVMappingFilterStringDefault)
};
}

#endif
