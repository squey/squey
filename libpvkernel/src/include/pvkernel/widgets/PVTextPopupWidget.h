
#ifndef PVWIDGETS_PVTEXTPOPUPWIDGET_H
#define PVWIDGETS_PVTEXTPOPUPWIDGET_H

#include <pvkernel/widgets/PVPopupWidget.h>

class QWebView;

namespace PVWidgets
{

/**
 * a class to text widget (as HTML content) over a QWidget
 *
 * @todo make this widget always on top of its parent to be
 * resized/moved with it
 */

class PVTextPopupWidget : public PVPopupWidget
{
public:
	/**
	 * a constructor
	 *
	 * @param parent the parent QWidget
	 */
	PVTextPopupWidget(QWidget* parent);

	/**
	 * set the content (which is HTML text)
	 *
	 * See http://www.developer.nokia.com/Community/Discussion/showthread.php?188112-QWebView-load-page-from-memory, for examples about how to set text.
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

private:
	QWebView* _webview;
};

}

#endif // PVWIDGETS_PVTEXTPOPUPWIDGET_H
