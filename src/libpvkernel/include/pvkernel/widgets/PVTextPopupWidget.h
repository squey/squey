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

#ifndef PVWIDGETS_PVTEXTPOPUPWIDGET_H
#define PVWIDGETS_PVTEXTPOPUPWIDGET_H

#include <pvkernel/widgets/PVPopupWidget.h>
#include <pvkernel/core/PVTheme.h>

#include <QString>

class QTextBrowser;
class QPaintEvent;

namespace PVWidgets
{

/**
 * a class to display a text widget (with HTML content) over a QWidget.
 *
 * The content uses a table based structure; with table, columns and text.
 *
 * A sample is worth a thousand words:
 *
 * \code
 * help->initTextFromFile("view help");
 * help->addTextFromFile(":help-table1-col1-ele1");
 * help->addTextFromFile(":help-table1-col1-ele2");
 * help->addTextFromFile(":help-table1-col1-ele3");
 * help->newColumn();
 * help->addTextFromFile(":help-table1-col2-ele1");
 * help->newTable();
 * help->addTextFromFile(":help-table2-col1-ele1");
 * help->newColumn();
 * help->addTextFromFile(":help-table2-col2-ele1");
 * help->addTextFromFile(":help-table2-col2-ele2");
 * help->finalizeText();
 * \endcode
 */

class PVTextPopupWidget : public PVPopupWidget
{
  public:
	/**
	 * a orizable enumeration to tell how a popup will be placed on screen
	 */
	typedef enum {
		AlignNone = 0,
		AlignLeft = 1,
		AlignRight = 2,
		AlignHCenter = 4,
		AlignTop = 8,
		AlignBottom = 16,
		AlignVCenter = 32,
		AlignUnderMouse = 64,
		AlignCenter = AlignHCenter + AlignVCenter
	} AlignEnum;

	/**
	 * a "orizable" enumeration to tell how a popup is expanded
	 */
	typedef enum {
		ExpandNone = 0,
		ExpandX = 1,
		ExpandY = 2,
		ExpandAll = ExpandX + ExpandY
	} ExpandEnum;

  public:
	/**
	 * a constructor
	 *
	 * @param parent the parent QWidget
	 */
	explicit PVTextPopupWidget(QWidget* parent);

	/**
	 * set the content (which is HTML text)
	 *
	 * See
	 *http://www.developer.nokia.com/Community/Discussion/showthread.php?188112-QWebView-load-page-from-memory,
	 *for examples about how to set text.
	 *
	 * @param text the HTML text to display
	 */
	void setText(const QString& text);

	/**
	 * set the content (which is HTML text) from a file
	 *
	 * @param filename the HTML file to display
	 */
	void setTextFromFile(const QString& filename);

	/**
	 * initialize the text content
	 *
	 * the columns use div html tags
	 *
	 * @param title the page's title
	 */
	void initTextFromFile(const QString& title);

	/**
	 * add a new entry in the current column
	 *
	 * The html_filename's content must be an p block followed by a table.
	 *
	 * @param html_filename the filename or resource id of the html content
	 */
	void addTextFromFile(const QString& html_filename);

	/**
	 * close the current column and open the next one
	 */
	void newColumn();

	/**
	 * close the current table and open a new one
	 */
	void newTable();

	/**
	 * close the help text and set it.
	 */
	void finalizeText();

	/**
	 * make the popup visible over a widget
	 *
	 * it remains shown until it is explictly hidden. I.e. if its parent
	 * widget is resized or moved, the popup's geometry will be updated
	 * according to its parent's change.
	 *
	 * @param widget the parent widget
	 * @param align how the popup must be aligned according to its parent
	 * @param align how the popup must be expanded according to its parent
	 * @param border the border around the popup in its parent's geometry
	 */
	void popup(QWidget* widget, int align = AlignNone, int expand = ExpandNone, int border = 0);

  protected:
	/**
	 * reimplement PVPopupWidget::setVisible(bool)
	 *
	 * to install/remove the eventfilter on the parent widget and maintain
	 * the focus on the parent widget.
	 */
	void setVisible(bool visible) override;

	/**
	 * reimplement QObject::eventFilter
	 *
	 * to reconfigure the popup it when its alignment widget's
	 * geometry has changed.
	 */
	bool eventFilter(QObject* obj, QEvent* event) override;

  private:
	/**
	 * slot called when the app color scheme has changed
	 */
	void refresh_theme(PVCore::PVTheme::EColorScheme cs);

	/**
	 * return the content of the CSS "style" tag based on the current theme color scheme
	 */
	QString get_style();

  private:
	QTextBrowser* _webview;
	QWidget* _last_widget;
	QString _temp_text;
	int _last_align;
	int _last_expand;
	int _last_border;
};
} // namespace PVWidgets

#endif // PVWIDGETS_PVTEXTPOPUPWIDGET_H
