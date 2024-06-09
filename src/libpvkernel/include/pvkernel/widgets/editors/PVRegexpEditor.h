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

#ifndef PVCORE_PVREGEXPEDITOR_H
#define PVCORE_PVREGEXPEDITOR_H

#include <qtmetamacros.h>
#include <QLineEdit>
#include <QRegExp>

class QWidget;

namespace PVWidgets
{

/**
 * \class PVRegexpEditor
 */
class PVRegexpEditor : public QLineEdit
{
	Q_OBJECT
	Q_PROPERTY(QRegExp _rx READ get_rx WRITE set_rx USER true)

  public:
	explicit PVRegexpEditor(QWidget* parent = nullptr);
	~PVRegexpEditor() override;

  public:
	QRegExp get_rx() const;
	void set_rx(QRegExp rx);
};
} // namespace PVWidgets

#endif
