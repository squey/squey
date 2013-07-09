
#include <pvkernel/widgets/PVTextPopupWidget.h>

#include <QVBoxLayout>
#include <QWebView>
#include <QWebPage>
#include <QFile>
#include <QResizeEvent>
#include <QKeyEvent>

#define AlignHoriMask (PVWidgets::PVTextPopupWidget::AlignLeft | PVWidgets::PVTextPopupWidget::AlignRight | PVWidgets::PVTextPopupWidget::AlignHCenter)
#define AlignVertMask (PVWidgets::PVTextPopupWidget::AlignTop | PVWidgets::PVTextPopupWidget::AlignBottom | PVWidgets::PVTextPopupWidget::AlignVCenter)

//#define DEFAULT_HTML_TEXT "<html><body style=\"background-color: transparent; -webkit-opacity: 0.5; color: white;\">default text</body></html>"
#define DEFAULT_HTML_TEXT "<html><body style=\"background-color: #FF0000; color: white;\">default text</body></html>"

static QRect reconfigure_geometry(const QRect current_geom, const QWidget* widget,
                                  int align, int expand, int border)
{
	QRect parent_geom = widget->geometry();

	/* about borders
	 */
	parent_geom = QRect(parent_geom.x() + border,
	                    parent_geom.y() + border,
	                    parent_geom.width() - 2 * border,
	                    parent_geom.height() - 2 * border);

	// parent_geom.moveTo(widget->mapToGlobal(parent_geom.topLeft()));

	QPoint center_pos = parent_geom.center();
	QRect new_geom;

 	if (expand & PVWidgets::PVTextPopupWidget::ExpandX) {
		new_geom.setWidth(parent_geom.width());
	} else {
		new_geom.setWidth(current_geom.width());
	}

	if (expand & PVWidgets::PVTextPopupWidget::ExpandY) {
		new_geom.setHeight(parent_geom.height());
	} else {
		new_geom.setHeight(current_geom.height());
	}

	switch(align & AlignHoriMask) {
	case PVWidgets::PVTextPopupWidget::AlignRight:
		new_geom.moveLeft(parent_geom.right() - new_geom.width());
		break;
	case PVWidgets::PVTextPopupWidget::AlignHCenter:
		new_geom.moveLeft(center_pos.x() - new_geom.width() / 2);
		break;
	case PVWidgets::PVTextPopupWidget::AlignLeft:
	default:
		new_geom.moveLeft(parent_geom.left());
		break;
	}
	switch(align & AlignVertMask) {
	case PVWidgets::PVTextPopupWidget::AlignBottom:
		new_geom.moveTop(parent_geom.bottom() - new_geom.height());
		break;
	case PVWidgets::PVTextPopupWidget::AlignVCenter:
		new_geom.moveTop(center_pos.y() - new_geom.height() / 2);
		break;
	case PVWidgets::PVTextPopupWidget::AlignTop:
	default:
		new_geom.moveTop(parent_geom.top());
		break;
	}

	return new_geom;
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::PVTextPopupWidget
 *****************************************************************************/

PVWidgets::PVTextPopupWidget::PVTextPopupWidget(QWidget* parent) :
	PVWidgets::PVPopupWidget::PVPopupWidget(parent),
	_last_widget(0)
{
	setWindowFlags(Qt::FramelessWindowHint);

	setFocusPolicy(Qt::NoFocus);

	QVBoxLayout* l = new QVBoxLayout();
	l->setContentsMargins(0, 0, 0, 0);
	setLayout(l);

	_webview = new QWebView();

	// no need for "reload" context menu
	_webview->setContextMenuPolicy(Qt::NoContextMenu);
	_webview->setHtml(DEFAULT_HTML_TEXT);
	l->addWidget(_webview);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::setText
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::setText(const QString& text)
{
	_webview->setHtml(text);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::setTextFromFile
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::setTextFromFile(const QString& filename)
{
	QFile file(filename);
	QString text(DEFAULT_HTML_TEXT);

	if (file.open(QIODevice::ReadOnly)) {
		QByteArray data;
		data = file.read(file.size());
		text = QString(data);
	} else {
		text = "<html><body style=\"background-color: #1F1F1F; webkit-opacity: 0.5; color: white;\">ERROR: file <b>" + filename + "</b> can not be loaded</body></html>";
	}

	setText(text);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::popup
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::popup(QWidget* widget,
                                         int align, int expand,
                                         int border)
{
	if (isVisible()) {
		return;
	}

	_last_widget = widget;
	_last_align = align;
	_last_expand = expand;
	_last_border = border;

	// make sure the popup's geometry is correct
	adjustSize();

	QRect new_geom = reconfigure_geometry(geometry(),
	                                      widget,
	                                      align, expand,
	                                      border);

	setGeometry(new_geom);
	raise();
	show();
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::setVisible
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::setVisible(bool visible)
{
	QWidget::setVisible(visible);

	if (visible) {
		parentWidget()->installEventFilter(this);
	} else {
		parentWidget()->removeEventFilter(this);
		_last_widget = nullptr;
	}

	if (_last_widget) {
		_last_widget->setFocus();
	}
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::eventFilter
 *****************************************************************************/

bool PVWidgets::PVTextPopupWidget::eventFilter(QObject *obj, QEvent *event)
{
	if (_last_widget == nullptr) {
		return false;
	}

	if (event->type() == QEvent::Resize) {
		 QRect geom = reconfigure_geometry(geometry(), _last_widget,
		                                   _last_align, _last_expand,
		                                   _last_border);

		 setGeometry(geom);

		 return false;
	} else if (event->type() == QEvent::KeyPress) {
		if (static_cast<QKeyEvent*>(event)->key() == Qt::Key_Escape) {
			hide();
			return false;
		}
	}
	// standard event processing
	return QObject::eventFilter(obj, event);
}
