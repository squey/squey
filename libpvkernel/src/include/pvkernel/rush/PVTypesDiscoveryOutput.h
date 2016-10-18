/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef __PVTYPESDISCOVERYOUTPUT_H__
#define __PVTYPESDISCOVERYOUTPUT_H__

#include <pvkernel/rush/PVControllerJob.h> // for PVControllerJob, etc
#include <pvkernel/rush/PVOutput.h>        // for PVOutput

#include <pvkernel/filter/PVFilterFunction.h> // for CLASS_FILTER_NONREG

#include <pvbase/types.h> // for PVRow

#include <cassert> // for assert

#include <pvcop/formatter_desc_list.h>

namespace pvcop
{
namespace types
{
class formatter_interface;
}
}

namespace PVRush
{

class PVTypesDiscoveryOutput : public PVRush::PVOutput
{
  public:
	using type_desc_t = std::pair<std::string, std::string>;

  public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	void operator()(PVCore::PVChunk* out) override;
	PVRow get_rows_count() override { return _row_count; }

  public:
	type_desc_t type_desc(PVCol col) const;

  protected:
	void prepare_load(const PVRush::PVFormat&) override;
	void job_has_finished(const PVControllerJob::invalid_elements_t&) override;

	CLASS_FILTER_NONREG(PVNrawTypesDiscovery)

  private:
	using matching_formatters_t = std::vector<std::vector<bool>>;
	using types_desc_t = std::vector<type_desc_t>;

	std::vector<pvcop::types::formatter_interface*> _formatters;
	types_desc_t _types_desc;
	matching_formatters_t _matching_formatters;
	size_t _column_count = 0;
	size_t _row_count = 0;
};
} // namespace PVRush

#endif // __PVTYPESDISCOVERYOUTPUT_H__
