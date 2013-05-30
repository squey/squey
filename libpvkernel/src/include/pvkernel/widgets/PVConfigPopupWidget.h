
#ifndef PVWIDGETS_PVCONFIGPOPUPWIDGET_H
#define PVWIDGETS_PVCONFIGPOPUPWIDGET_H

#include <pvkernel/widgets/PVPopupWidget.h>

class QWidget;
class QLayout;
class QPushButton;
class QToolButton;

namespace PVWidgets
{

/**
 * a class to display a configuration widget over a QWidget
 *
 * @todo: create a custom title-bar?
 */

class PVConfigPopupWidget : public PVPopupWidget
{
Q_OBJECT

public:
	/**
	 * a constructor
	 *
	 * @param parent the parent QWidget
	 * @param persistent true if the popup must be intially in persistent mode; false otherwise
	 */
	PVConfigPopupWidget(QWidget* parent, bool persistent = false);

	/**
	 * tell if the popup is persistent or not
	 *
	 * @return true is the popup is persistent; false otherwise.
	 */
	bool persistence() const;

	/**
	 * set the layout used for your content
	 *
	 * @note: do not use QWidget::setLayout(...) !
	 */
	void setContentLayout(QLayout* l);

public slots:
	/**
	 * set if the popup is persistent or not
	 *
	 * A persistent popup is a
	 * @return true is the popup is persistent; false otherwise.
	 */
	void setPersistence(bool persistent);

private:
	QToolButton* _pers_button;
};

}

#endif // PVWIDGETS_PVCONFIGPOPUPWIDGET_H
