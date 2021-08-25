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

#ifndef PVPARALLELVIEW_PVFULLPARALLELVIEWSELECTIONSQUARE_H
#define PVPARALLELVIEW_PVFULLPARALLELVIEWSELECTIONSQUARE_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVSelectionRectangle.h>

namespace PVParallelView
{

class PVFullParallelScene;
class PVLinesView;

class PVFullParallelViewSelectionRectangle : public PVSelectionRectangle
{
  public:
	struct barycenter {
		size_t zone_index1;
		size_t zone_index2;
		double factor1;
		double factor2;

		barycenter() { clear(); }

		void clear()
		{
			zone_index1 = PVZONEINDEX_INVALID;
			zone_index2 = PVZONEINDEX_INVALID;
			factor1 = 0.0;
			factor2 = 0.0;
		}
	};

  public:
	explicit PVFullParallelViewSelectionRectangle(PVFullParallelScene* fps);

  public:
	void clear() override;

  public:
	void update_position();

  protected:
	void commit(bool use_selection_modifiers) override;

  private:
	void store();

	PVFullParallelScene* scene_parent();
	PVFullParallelScene const* scene_parent() const;

	PVLinesView const& get_lines_view() const;

  private:
	PVFullParallelScene* _fps;
	barycenter _barycenter;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVFULLPARALLELVIEWSELECTIONSQUARE_H
