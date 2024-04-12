/* * MIT License
 *
 * © Squey, 2024
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

#ifndef __PVKERNEL_PVPARQUETEXPORTERWIDGET_H__
#define __PVKERNEL_PVPARQUETEXPORTERWIDGET_H__

#include <pvkernel/widgets/PVExporterWidgetInterface.h>
#include <pvkernel/rush/PVInputType.h>

namespace PVRush
{
	class PVParquetBase;
}

namespace PVWidgets
{

class PVParquetExporterWidget : public PVWidgets::PVExporterWidgetInterface
{
	Q_OBJECT

  public:
	PVParquetExporterWidget(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw);
	virtual ~PVParquetExporterWidget(){};

  public:
	PVRush::PVExporterBase& exporter() override { return *_exporter; }

  private:
	std::unique_ptr<PVRush::PVExporterBase> _exporter;
};

} // namespace PVWidgets

#endif // __PVKERNEL_PVPARQUETEXPORTERWIDGET_H__
