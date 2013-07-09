
#ifndef PVWIDGETS_PVTEXTPOPUPWIDGET_H
#define PVWIDGETS_PVTEXTPOPUPWIDGET_H

#include <pvkernel/widgets/PVPopupWidget.h>

class QWebView;
class QPaintEvent;

namespace PVWidgets
{

/**
 * a class to text widget (as HTML content) over a QWidget
 *
 * @todo make this widget always on top of its parent to be
 * resized/moved with it
 * @todo make this widget transparent
 * @todo make text unselectable (work-around in html's css:
 * -webkit-user-select: none;
 */

class PVTextPopupWidget : public PVPopupWidget
{
public:
	/**
	 * a orizable enumeration to tell how a popup will be placed on screen
	 */
	typedef enum {
		AlignNone       =  0,
		AlignLeft       =  1,
		AlignRight      =  2,
		AlignHCenter    =  4,
		AlignTop        =  8,
		AlignBottom     = 16,
		AlignVCenter    = 32,
		AlignUnderMouse = 64,
		AlignCenter     = AlignHCenter + AlignVCenter
	} AlignEnum;

	/**
	 * a "orizable" enumeration to tell how a popup is expanded
	 */
	typedef enum {
		ExpandNone = 0,
		ExpandX    = 1,
		ExpandY    = 2,
		ExpandAll  = ExpandX + ExpandY
	} ExpandEnum;

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
	void popup(QWidget* widget, int align = AlignNone, int expand = ExpandNone,
	           int border = 0);

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
	bool eventFilter(QObject *obj, QEvent *event);

private:
	QWebView* _webview;
	QWidget*  _last_widget;
	int       _last_align;
	int       _last_expand;
	int       _last_border;
};

}

#endif // PVWIDGETS_PVTEXTPOPUPWIDGET_H
