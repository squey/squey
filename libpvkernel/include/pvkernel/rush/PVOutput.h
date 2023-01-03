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

#ifndef PVOUTPUT_FILE_H
#define PVOUTPUT_FILE_H

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <atomic>

namespace PVRush
{

class PVFormat;
class PVControllerJob;

class PVOutput : public PVFilter::PVFilterFunctionBase<void, PVCore::PVTextChunk*>
{
	friend class PVControllerJob;

  public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	virtual void operator()(PVCore::PVChunk* out) = 0;

  public:
	virtual PVRow get_rows_count() = 0;
	size_t get_out_size() const { return _out_size; }

  public:
	virtual void prepare_load(const PVRush::PVFormat&){};

  protected:
	// This function is called by PVControllerJob
	// when its job has finished.
	virtual void job_has_finished(const std::map<size_t, std::string>&) {}

	CLASS_FILTER_NONREG(PVOutput)

  protected:
	std::atomic<size_t> _out_size{
	    0}; //!< Total size handled by the pipeline. (metrics depend on inputs)
};
} // namespace PVRush

#endif
