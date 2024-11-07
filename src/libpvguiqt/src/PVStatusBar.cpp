#include <pvguiqt/PVStatusBar.h>

#include <pvkernel/core/PVWSLHelper.h>
#include <pvparallelview/PVParallelView.h>

#include <QStyle>

PVGuiQt::PVStatusBar::PVStatusBar(QWidget* parent /*= nullptr*/)
    : QStatusBar(parent)
    , _default_mouse_buttons_legend()
{
    QWidget* mouse_widget = new QWidget;
    mouse_widget->setStyleSheet("background-color: transparent;");
    QHBoxLayout* mouse_widget_layout = new QHBoxLayout(mouse_widget);
    mouse_widget_layout->setContentsMargins(0, 0, 0, 0);

    const size_t LEGEND_WIDTH = 150;

    // Mouse left button
    PVModdedIconLabel* mouse_left_label_icon = new PVModdedIconLabel("mouse-left-button", QSize(16, 16));
    _mouse_left_label_text = new QLabel();
    _mouse_left_label_text->setFixedWidth(LEGEND_WIDTH);
    mouse_widget_layout->addWidget(mouse_left_label_icon);
    mouse_widget_layout->addWidget(_mouse_left_label_text);

    // Mouse scrollwheel
    PVModdedIconLabel* mouse_scrollwheel_label_icon = new PVModdedIconLabel("computer-mouse-scrollwheel", QSize(16, 16));
    _mouse_scrollwheel_label_text = new QLabel();
    _mouse_scrollwheel_label_text->setFixedWidth(LEGEND_WIDTH);
    mouse_widget_layout->addWidget(mouse_scrollwheel_label_icon);
    mouse_widget_layout->addWidget(_mouse_scrollwheel_label_text);

    // // Mouse right button
    PVModdedIconLabel* mouse_right_label_icon = new PVModdedIconLabel("mouse-right-button", QSize(16, 16));
    _mouse_right_label_text = new QLabel();
    _mouse_right_label_text->setFixedWidth(LEGEND_WIDTH);
    mouse_widget_layout->addWidget(mouse_right_label_icon);
    mouse_widget_layout->addWidget(_mouse_right_label_text);

    addPermanentWidget(mouse_widget);

    QHBoxLayout* right_layout = new QHBoxLayout();
    right_layout->addStretch(1);

	/**
	 * Show warning message when no GPU accelerated device has been found
	 * Except under WSL where GPU is not supported yet
	 * (https://wpdev.uservoice.com/forums/266908-command-prompt-console-windows-subsystem-for-l/suggestions/16108045-opencl-cuda-gpu-support)
	 */
	if (/*not PVParallelView::common::is_gpu_accelerated()*/ false and not PVCore::PVWSLHelper::is_microsoft_wsl()) {
		QIcon warning_icon = QApplication::style()->standardIcon(QStyle::SP_MessageBoxWarning);
		auto* warning_label_icon = new QLabel;
		warning_label_icon->setPixmap(warning_icon.pixmap(QSize(16, 16)));
		right_layout->addWidget(warning_label_icon);

        _warning_msg = new QLabel();
        set_gpu_warning_message(PVCore::PVTheme::color_scheme());
        connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this, &PVGuiQt::PVStatusBar::set_gpu_warning_message);

		right_layout->addWidget(_warning_msg);
	}

    // version widget
    right_layout->setContentsMargins(0, 0, 0, 0);
    QLabel* version_label = new QLabel(QString(SQUEY_CURRENT_VERSION_STR));
    right_layout->addWidget(version_label);
    QWidget* version_widget = new QWidget;
    version_widget->setStyleSheet("background-color: transparent;");
    version_widget->setLayout(right_layout);
    addPermanentWidget(version_widget);

    set_mouse_buttons_legend(_default_mouse_buttons_legend);
}

void PVGuiQt::PVStatusBar::set_mouse_buttons_legend(const PVWidgets::PVMouseButtonsLegend& legend)
{
    set_mouse_left_button_legend(legend.left_button_legend());
    set_mouse_right_button_legend(legend.right_button_legend());
    set_mouse_scrollwheel_legend(legend.scrollwheel_button_legend());
}

void PVGuiQt::PVStatusBar::clear_mouse_buttons_legend()
{
    set_mouse_buttons_legend(_default_mouse_buttons_legend);
}

void PVGuiQt::PVStatusBar::set_mouse_left_button_legend(QString text)
{
    _mouse_left_label_text->setText(text);
}

void PVGuiQt::PVStatusBar::set_mouse_right_button_legend(QString text)
{
    _mouse_right_label_text->setText(text);
}

void PVGuiQt::PVStatusBar::set_mouse_scrollwheel_legend(QString text)
{
    _mouse_scrollwheel_label_text->setText(text);
}

void PVGuiQt::PVStatusBar::set_gpu_warning_message(PVCore::PVTheme::EColorScheme cs)
{
    const QColor& color = (cs == PVCore::PVTheme::EColorScheme::LIGHT ? 0xfa6000 : 0xdf920c);
    assert(_warning_msg);
	_warning_msg->setText("<p align=\"right\"><font color=\"" + color.name() + "\"><b>You are running in degraded "
		                  "mode without GPU acceleration. </b></font></p>");
}