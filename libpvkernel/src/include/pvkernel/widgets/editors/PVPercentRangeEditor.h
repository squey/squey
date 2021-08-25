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

#ifndef PVWIDGETS_PVPERCENTRANGEEDITOR_H
#define PVWIDGETS_PVPERCENTRANGEEDITOR_H

#include <pvkernel/widgets/PVAbstractRangePicker.h>

#include <pvkernel/core/PVPercentRangeType.h>

class QWidget;

namespace PVWidgets
{

/**
 * @class PVPercentRangeEditor
 */
class PVPercentRangeEditor : public PVAbstractRangePicker
{
	Q_OBJECT

	Q_PROPERTY(
	    PVCore::PVPercentRangeType _percent_range_type READ get_values WRITE set_values USER true)

  public:
	/**
	 * Constructor
	 */
	explicit PVPercentRangeEditor(QWidget* parent = nullptr);

  public:
	/**
	 * returns the editor's curent value
	 *
	 * @return the percent pair
	 */
	PVCore::PVPercentRangeType get_values() const;

	/**
	 * set the editor's curent value
	 *
	 * @param r the value to use
	 */
	void set_values(const PVCore::PVPercentRangeType& r);
};
} // namespace PVWidgets

#endif // PVWIDGETS_PVPERCENTRANGEEDITOR_H
