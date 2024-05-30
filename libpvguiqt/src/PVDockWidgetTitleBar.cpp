#include <pvguiqt/PVDockWidgetTitleBar.h>

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QKeyEvent>
#include <QDockWidget>

#include <pvkernel/core/PVTheme.h>

PVGuiQt::PVDockWidgetTitleBar::PVDockWidgetTitleBar(Squey::PVView* view, QWidget* view_widget, bool has_help_page, QWidget* parent /* = nullptr */)
    : QWidget(parent)
{
    // Setup custom title widget to handle help button
	const QSize icon_size(12, 12);
	QHBoxLayout* layout = new QHBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);

	QPixmap color_pixmap(16, 16);
	color_pixmap.fill(view->get_color());
	QLabel* color_square = new QLabel();
	color_square->setPixmap(color_pixmap);
	layout->addWidget(color_square);
	_window_title = new QLabel(view_widget->windowTitle());
	_window_title->setStyleSheet("font-weight: bold;");
	QWidget* color_title = new QWidget();
	QHBoxLayout* color_title_layout = new QHBoxLayout(color_title);
	color_title_layout->setContentsMargins(0, 0, 0, 0);
	color_title_layout->setSpacing(0);
	color_title_layout->addWidget(color_square);
	color_title_layout->addWidget(_window_title);
	layout->addWidget(color_title);
	layout->addStretch(1);

	// Handle widget title changes
	connect(view_widget, &QWidget::windowTitleChanged, [=] { _window_title->setText(view_widget->windowTitle()); });

	// Help button
	_help_button = new QPushButton();
	_help_button->setCheckable(true);
	_help_button->setFixedSize(icon_size);
	set_help_button_stylesheet();
	connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this, &PVGuiQt::PVDockWidgetTitleBar::set_help_button_stylesheet);
	layout->addWidget(_help_button);
	connect(_help_button, &QPushButton::clicked, parentWidget(), [this](){
		QKeyEvent* keypress_event  = new QKeyEvent (QEvent::KeyPress, Qt::Key_Help, Qt::NoModifier, "?");
		qApp->postEvent((QObject*)qobject_cast<QDockWidget*>(parentWidget())->widget(),(QEvent *)keypress_event);
	});
	set_help_page_visible(has_help_page);

	// Float button
	_float_button = new QPushButton();
	_float_button->setFixedSize(icon_size);
	set_float_button_stylesheet();
	connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this, &PVGuiQt::PVDockWidgetTitleBar::set_float_button_stylesheet);
	connect(_float_button, &QPushButton::clicked, parentWidget(), [this](){qobject_cast<QDockWidget*>(parentWidget())->setFloating(not qobject_cast<QDockWidget*>(parentWidget())->isFloating());});
	layout->addWidget(_float_button);

	// Close button
	_close_button = new QPushButton();
	_close_button->setFixedSize(icon_size);
	set_close_button_stylesheet();
	connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this, &PVGuiQt::PVDockWidgetTitleBar::set_close_button_stylesheet);
    connect(_close_button, &QPushButton::clicked, parentWidget(), [this](){parentWidget()->close();});
	layout->addWidget(_close_button);
}

void PVGuiQt::PVDockWidgetTitleBar::set_help_page_visible(bool visible)
{
	_help_button->setVisible(visible);
}

void PVGuiQt::PVDockWidgetTitleBar::mouseMoveEvent(QMouseEvent* event) 
{
	event->ignore();
	QWidget::mouseMoveEvent(event);
}

void PVGuiQt::PVDockWidgetTitleBar::set_window_title(const QString& window_title)
{
	_window_title->setText(window_title);
}

void PVGuiQt::PVDockWidgetTitleBar::set_help_button_stylesheet()
{
	set_button_stylesheet(_help_button, "question");
}

void PVGuiQt::PVDockWidgetTitleBar::set_float_button_stylesheet()
{
	set_button_stylesheet(_float_button, "window-undock");
}

void PVGuiQt::PVDockWidgetTitleBar::set_close_button_stylesheet()
{
	set_button_stylesheet(_close_button, "window-close");
}

void PVGuiQt::PVDockWidgetTitleBar::set_button_stylesheet(QPushButton* button, const QString& icon_name)
{
	const QString& theme = PVCore::PVTheme::color_scheme_name();
	button->setStyleSheet(QString(
		"*{background-color: transparent; border-image: url(:/qss_icons/%1/rc.%1/%2.png);}"
		":hover{ border-image: url(:/qss_icons/%1/rc.%1/%2_focus.png);}"
		":pressed{ border-image: url(:/qss_icons/%1/rc.%1/%2_pressed.png);}"
		":checked{ border-image: url(:/qss_icons/%1/rc.%1/%2_pressed.png);}"
		).arg(theme, icon_name)
	);
}
