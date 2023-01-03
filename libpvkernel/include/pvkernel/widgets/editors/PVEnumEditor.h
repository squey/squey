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

#ifndef PVCORE_PVENUMEDITOR_H
#define PVCORE_PVENUMEDITOR_H

#include <pvkernel/core/PVEnumType.h>

#include <QComboBox>

class QWidget;

namespace PVWidgets
{

/**
 * \class PVEnumEditor
 */
class PVEnumEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVEnumType _enum READ get_enum WRITE set_enum USER true)

  public:
	using QComboBox::QComboBox;

  public:
	PVCore::PVEnumType get_enum() const;
	void set_enum(PVCore::PVEnumType e);

  protected:
	PVCore::PVEnumType _e;
};
} // namespace PVWidgets

#endif // PVCORE_PVEnumEDITOR_H
