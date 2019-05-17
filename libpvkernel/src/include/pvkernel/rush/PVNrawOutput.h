/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVNRAWOUTPUT_FILE_H
#define PVNRAWOUTPUT_FILE_H

#include <pvkernel/rush/PVControllerJob.h> // for PVControllerJob, etc
#include <pvkernel/rush/PVOutput.h>        // for PVOutput

#include <pvkernel/filter/PVFilterFunction.h> // for CLASS_FILTER_NONREG

#include <pvbase/types.h> // for PVRow

#include <cassert> // for assert

namespace PVCore
{
class PVTextChunk;
} // namespace PVCore
namespace PVRush
{
class PVFormat;
class PVNraw;
} // namespace PVRush

namespace PVRush
{

class PVNrawOutput : public PVRush::PVOutput
{
  public:
	explicit PVNrawOutput(PVNraw& nraw);
	PVNrawOutput() = delete;

  public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVTextChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	void operator()(PVCore::PVTextChunk* out) override;

	void set_nraw_dest(PVNraw& nraw) { _nraw_dest = &nraw; }

  public:
	PVRow get_rows_count() override;

  public:
	PVNraw const& nraw_dest() const
	{
		assert(_nraw_dest);
		return *_nraw_dest;
	}
	PVNraw& nraw_dest()
	{
		assert(_nraw_dest);
		return *_nraw_dest;
	}

  protected:
	void prepare_load(const PVRush::PVFormat& format) override;
	void job_has_finished(const PVControllerJob::invalid_elements_t& inv_elts) override;

  protected:
	PVNraw* _nraw_dest;

	CLASS_FILTER_NONREG(PVNrawOutput)
};
} // namespace PVRush

#endif
