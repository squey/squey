
#include <pvkernel/widgets/PVHelpWidget.h>

#include <QtCore> // for Qt::Key_*

/*****************************************************************************
 * PVWidgets::PVHelpWidget::PVHelpWidget
 *****************************************************************************/

PVWidgets::PVHelpWidget::PVHelpWidget(QWidget *parent) :
	PVWidgets::PVTextPopupWidget(parent)
{}

/*****************************************************************************
 * PVWidgets::PVHelpWidget::is_close_key
 *****************************************************************************/

bool PVWidgets::PVHelpWidget::is_close_key(int key)
{
	if (is_help_key(key)) {
		return true;
	}

	return PVPopupWidget::is_close_key(key);
}

/*****************************************************************************
 * PVWidgets::PVHelpWidget::is_help_key
 *****************************************************************************/

bool PVWidgets::PVHelpWidget::is_help_key(int key)
{
	return ((key == Qt::Key_Question)
	        || (key == Qt::Key_Help)
	        || (key == Qt::Key_F1));
}
