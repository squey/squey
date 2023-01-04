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

#ifndef PVCORE_PVORIGINALAXISINDEXEDITOR_H
#define PVCORE_PVORIGINALAXISINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <inendi/PVView.h>

namespace PVWidgets
{

/**
 * \class PVOriginalAxisIndexEditor
 */
class PVOriginalAxisIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVOriginalAxisIndexType _axis_index READ get_axis_index WRITE set_axis_index
	               USER true)

  public:
	explicit PVOriginalAxisIndexEditor(Inendi::PVView const& view, QWidget* parent = 0);

  public:
	PVCore::PVOriginalAxisIndexType get_axis_index() const;
	void set_axis_index(PVCore::PVOriginalAxisIndexType axis_index);

  protected:
	Inendi::PVView const& _view;
};
} // namespace PVWidgets

#endif // PVCORE_PVAXISINDEXEDITOR_H
