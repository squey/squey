
#ifndef PVWIDGETS_PVPOPUPWIDGET_H
#define PVWIDGETS_PVPOPUPWIDGET_H

#include <QDialog>

namespace PVWidgets
{

/**
 * a generic popup widget to display any widget set over an QWidget
 *
 * @note  it may be useful to distinguish the parent widget which is used to
 * align the PVPopupWidget and the parent widget to give the focus to make
 * shortcuts work...
 */

class PVPopupWidget : public QDialog
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
	 * a orizable enumeration to tell how a popup is expanded
	 */
	typedef enum {
		ExpandNone = 0,
		ExpandX    = 1,
		ExpandY    = 2,
		ExpandAll  = ExpandX + ExpandY
	} ExpandEnum;

public:
	/**
	 * create a new popup widget
	 *
	 * @note contrary to Qt's widgets, a parent is required
	 *
	 * @param parent the parent QWidget
	 */
	PVPopupWidget(QWidget* parent);

public:
	/**
	 * make the popup visible at screen coord

	 */
	void popup(const QPoint& p, bool centered = false);

	void popup(QWidget* widget, int align = AlignNone, int expand = ExpandNone,
	           int border = 0, bool fit_in_screen = false);

	/**
	 * reimplement QDialog::setVisible(bool)
	 *
	 * to move focus from parent to poup and from popup to parent
	 */
	void setVisible(bool visible) override;
};

}

#endif // PVWIDGETS_PVPOPUPWIDGET_H
