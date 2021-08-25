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

#ifndef PVCORE_PVSPINBOXEDITOR_H
#define PVCORE_PVSPINBOXEDITOR_H

#include <QSpinBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/PVSpinBoxType.h>

namespace Inendi
{
class PVView;
} // namespace Inendi

namespace PVWidgets
{

class PVViewRowsSpinBoxEditor : public QSpinBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVSpinBoxType _s READ get_spin WRITE set_spin USER true)

  public:
	explicit PVViewRowsSpinBoxEditor(Inendi::PVView const& view, QWidget* parent = 0);
	~PVViewRowsSpinBoxEditor() override;

  public:
	PVCore::PVSpinBoxType get_spin() const;
	void set_spin(PVCore::PVSpinBoxType s);

  protected:
	PVCore::PVSpinBoxType _s;
	Inendi::PVView const& _view;
};
} // namespace PVWidgets

#endif // PVCORE_PVSPINBOXEDITOR_H
