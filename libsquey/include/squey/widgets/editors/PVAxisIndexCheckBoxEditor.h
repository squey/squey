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

#ifndef PVCORE_PVAXISINDEXCHECKBOXEDITOR_H
#define PVCORE_PVAXISINDEXCHECKBOXEDITOR_H

#include <QComboBox>
#include <QCheckBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/PVAxisIndexCheckBoxType.h>

#include <squey/PVView.h>

namespace PVWidgets
{

class PVAxisIndexCheckBoxEditor : public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVAxisIndexCheckBoxType _axis_index READ get_axis_index WRITE set_axis_index
	               USER true)

  private:
	QComboBox* combobox;
	QCheckBox* checkbox;
	bool _checked;
	int _current_index;

  public:
	explicit PVAxisIndexCheckBoxEditor(Squey::PVView const& view, QWidget* parent = nullptr);
	~PVAxisIndexCheckBoxEditor() override;

	PVCore::PVAxisIndexCheckBoxType get_axis_index() const;
	void set_axis_index(PVCore::PVAxisIndexCheckBoxType axis_index);

  protected:
	Squey::PVView const& _view;
};
} // namespace PVWidgets

#endif // PVCORE_PVAXISINDEXCHECKBOXEDITOR_H
