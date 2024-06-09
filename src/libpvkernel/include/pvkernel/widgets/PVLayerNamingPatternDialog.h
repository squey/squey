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

#ifndef PVWIDGETS_PVLAYERNAMINGPATTERNDIALOG_H
#define PVWIDGETS_PVLAYERNAMINGPATTERNDIALOG_H

#include <qstring.h>
#include <QDialog>

class QLineEdit;
class QComboBox;
class QWidget;

namespace PVWidgets
{

/**
 * This dialog is used to get a name for a (or more) layer, given a format string
 *
 *
 *
 */
class PVLayerNamingPatternDialog : public QDialog
{
  public:
	enum insert_mode { ON_TOP = 0, ABOVE_CURRENT, BELOW_CURRENT };

  public:
	/**
	 * CTOR
	 *
	 * @param title the dialog's title
	 * @param text the beginning of the explanation text
	 * @param pattern the default pattern to use
	 * @param m the default mode
	 * @param parent the parent widget
	 */
	PVLayerNamingPatternDialog(const QString& title,
	                           const QString& text,
	                           const QString& pattern = "%v",
	                           insert_mode m = ON_TOP,
	                           QWidget* parent = nullptr);

	/**
	 * get the pattern to use for layers' name(s)
	 *
	 * @return the pattern to use
	 */
	QString get_name_pattern() const;

	/**
	 * get the insertion mode to use
	 *
	 * @return the insertion mode to use
	 */
	insert_mode get_insertion_mode() const;

  private:
	QLineEdit* _line_edit;
	QComboBox* _combo_box;
};
} // namespace PVWidgets

#endif // PVWIDGETS_PVLAYERNAMINGPATTERNDIALOG_H
