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

#ifndef PVSCATTERVIEWDATAIMPL_H_
#define PVSCATTERVIEWDATAIMPL_H_

#include <pvparallelview/PVScatterViewDataInterface.h>

#include <pvkernel/core/PVMemory2D.h>

namespace PVParallelView
{

class PVScatterViewDataImpl : public PVScatterViewDataInterface
{
  public:
	void process_bg(ProcessParams const& params,
	                PVScatterViewImage& image,
	                tbb::task_group_context* ctxt = nullptr) const override;
	void process_sel(ProcessParams const& params,
	                 PVScatterViewImage& image,
	                 Squey::PVSelection const& sel,
	                 tbb::task_group_context* ctxt = nullptr) const override;

  private:
	static void process_image(ProcessParams const& params,
	                          PVScatterViewImage& image,
	                          Squey::PVSelection const* sel = nullptr,
	                          tbb::task_group_context* ctxt = nullptr);
};
} // namespace PVParallelView

#endif // PVSCATTERVIEWDATAIMPL_H_
