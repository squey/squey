/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVNRAWOUTPUT_FILE_H
#define PVNRAWOUTPUT_FILE_H

#include <pvbase/types.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/rush/PVNraw.h>

namespace PVRush
{

class PVNrawOutput : public PVRush::PVOutput
{
  public:
	PVNrawOutput(PVNraw& nraw);
	PVNrawOutput() = delete;

  public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	void operator()(PVCore::PVChunk* out) override;

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
	void job_has_finished() override;

  protected:
	PVNraw* _nraw_dest;

	CLASS_FILTER_NONREG(PVNrawOutput)
};
}

#endif
