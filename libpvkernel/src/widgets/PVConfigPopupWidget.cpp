
#include <pvkernel/widgets/PVConfigPopupWidget.h>

#include <QApplication>
#include <QDesktopWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QToolButton>
#include <QSpacerItem>

/*****************************************************************************
 * PVWidgets::PVConfigPopupWidget::PVConfigPopupWidget
 *****************************************************************************/

PVWidgets::PVConfigPopupWidget::PVConfigPopupWidget(QWidget* parent,
                                                    bool persistent) :
	PVWidgets::PVPopupWidget::PVPopupWidget(parent)
{
	QVBoxLayout* vbox = new QVBoxLayout();
	vbox->setContentsMargins(0, 0, 0, 0);
	setLayout(vbox);

	QHBoxLayout* hbox = new QHBoxLayout();
	hbox->setContentsMargins(0, 0, 0, 0);
	vbox->addLayout(hbox);

	QSpacerItem* s = new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed);
	hbox->addSpacerItem(s);

	_pers_button = new QToolButton();
	_pers_button->setCheckable(true);
	_pers_button->setFixedSize(16, 16);

	connect(_pers_button, SIGNAL(toggled(bool)),
	        this, SLOT(setPersistence(bool)));
	hbox->addWidget(_pers_button);

	setPersistence(persistent);
}

/*****************************************************************************
 * PVWidgets::PVConfigPopupWidget::setPersistence
 *****************************************************************************/

void PVWidgets::PVConfigPopupWidget::setPersistence(bool persistent)
{
	if (persistent != persistence()) {
		Qt::WindowFlags flags = 0;
		bool visible = isVisible();
		QRect geom;

		if (visible) {
			/* if the popup is already visible, its geometry has
			 * to be saved to make the popup's content's position
			 * fixed on the screen whatever the window type change
			 */
			geom = geometry();
			geom.moveTo(mapToGlobal(QPoint()));
		}

		if (persistent) {
			flags = Qt::Tool;
			_pers_button->setIcon(QIcon(":/pin-on"));
		} else {
			flags = Qt::Popup | Qt::FramelessWindowHint;
			_pers_button->setIcon(QIcon(":/pin-off"));
		}
		setWindowFlags(flags);

		_pers_button->blockSignals(true);
		_pers_button->setChecked(persistent);
		_pers_button->blockSignals(false);

		if (visible) {
			setGeometry(geom);

			/* calling setWindowFlags may hide the popup, so that,
			 * this state has to survive the type change.
			 */
			show();
		}
	}
}

/*****************************************************************************
 * PVWidgets::PVConfigPopupWidget::persistence
 *****************************************************************************/

bool PVWidgets::PVConfigPopupWidget::persistence() const
{
	return windowType() != Qt::Popup;
}

void PVWidgets::PVConfigPopupWidget::setContentLayout(QLayout* l)
{
	QVBoxLayout* vbox = static_cast<QVBoxLayout*>(layout());

	vbox->addLayout(l);
}
