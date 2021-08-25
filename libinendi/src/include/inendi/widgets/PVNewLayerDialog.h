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

#ifndef __PVGUIQT_PVNEWLAYERDIALOG_H__
#define __PVGUIQT_PVNEWLAYERDIALOG_H__

#include <QCheckBox>
#include <QDialog>
#include <QLineEdit>
#include <QMainWindow>
#include <QString>
#include <QWidget>

namespace PVWidgets
{

class PVNewLayerDialog : public QDialog
{
	Q_OBJECT;

  public:
	static QString get_new_layer_name_from_dialog(const QString& layer_name,
	                                              bool& hide_layers,
	                                              QWidget* parent_widget = nullptr);

  private:
	explicit PVNewLayerDialog(const QString& layer_name,
	                          bool hide_layers,
	                          QWidget* parent = nullptr);
	QString get_layer_name() const;
	bool should_hide_layers() const;

  private:
	QLineEdit* _text;
	QCheckBox* _checkbox;
};
} // namespace PVWidgets

#endif // __PVGUIQT_PVNEWLAYERDIALOG_H__
