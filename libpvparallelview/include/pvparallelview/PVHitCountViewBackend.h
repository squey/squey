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

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWBACKEND_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWBACKEND_H

#include <inendi/PVPlottedNrawCache.h>

#include <pvparallelview/PVHitGraphBlocksManager.h>

namespace Inendi
{
class PVView;
class PVPlottedNrawCache;
} // namespace Inendi

namespace PVParallelView
{

class PVHitCountViewBackend
{
  public:
	PVHitCountViewBackend(const Inendi::PVView& view, const PVCol axis);

	Inendi::PVPlottedNrawCache& get_y_labels_cache() { return _y_labels_cache; }
	PVHitGraphBlocksManager& get_hit_graph_manager() { return _hit_graph_manager; }

  private:
	Inendi::PVPlottedNrawCache _y_labels_cache;
	PVHitGraphBlocksManager _hit_graph_manager;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWBACKEND_H
