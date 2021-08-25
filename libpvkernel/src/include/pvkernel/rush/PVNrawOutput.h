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
	void prepare_load(const PVRush::PVFormat& format) override;
	void job_has_finished(const PVControllerJob::invalid_elements_t& inv_elts) override;

  protected:
	PVNraw* _nraw_dest;

	CLASS_FILTER_NONREG(PVNrawOutput)
};
} // namespace PVRush

#endif
