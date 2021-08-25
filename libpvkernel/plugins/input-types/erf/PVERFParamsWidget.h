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

#ifndef __PVERFPARAMSWIDGET_H__
#define __PVERFPARAMSWIDGET_H__

#include <QDialog>

#include <pvkernel/rush/PVFormat.h>

#include "PVERFTreeModel.h"
#include "../../common/erf/PVERFAPI.h"

namespace PVRush
{

class PVInputTypeERF;

class PVERFParamsWidget : public QDialog
{
	Q_OBJECT

  public:
	PVERFParamsWidget(PVInputTypeERF const* in_t, QWidget* parent);

  public:
	std::vector<QDomDocument> get_formats();
	QStringList paths() const { return _paths; }
	std::vector<std::tuple<rapidjson::Document, std::string, PVRush::PVFormat>>
	get_sources_info() const;

  private:
	std::unique_ptr<PVRush::PVERFTreeModel> _model;
	QStringList _paths;
	std::unique_ptr<PVERFAPI> _erf;
	bool _status_bar_needs_refresh;
};

} // namespace PVRush

#endif // __PVERFPARAMSWIDGET_H__
