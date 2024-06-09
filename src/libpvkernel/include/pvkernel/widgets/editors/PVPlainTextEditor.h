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

#ifndef PVCORE_PVPLAINTEXTEDITOR_H
#define PVCORE_PVPLAINTEXTEDITOR_H

#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <qtmetamacros.h>
#include <QWidget>
#include <QPlainTextEdit>

class QEvent;
class QObject;
class QPlainTextEdit;

namespace PVWidgets
{

/**
 * \class PVRegexpEditor
 */
class PVPlainTextEditor : public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVPlainTextType _text READ get_text WRITE set_text USER true)

  public:
	explicit PVPlainTextEditor(QWidget* parent = nullptr);

  public:
	PVCore::PVPlainTextType get_text() const;
	void set_text(PVCore::PVPlainTextType const& text);

  protected:
	bool eventFilter(QObject* object, QEvent* event) override;
	void save_to_file(bool append);

  protected Q_SLOTS:
	void slot_import_file();
	void slot_export_file();
	void slot_export_and_import_file();

  protected:
	QPlainTextEdit* _text_edit;

  private:
	PVWidgets::PVFileDialog _file_dlg;
};
} // namespace PVWidgets

#endif
