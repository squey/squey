//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <pvguiqt/PVImportPage.h>

#include <QFont>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLayoutItem>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QStyle>
#include <QVBoxLayout>

#include <algorithm>
#include <limits>
#include <utility>

namespace
{

QStyle::StandardPixmap pixmap_of(PVGuiQt::PVImportPage::Severity severity)
{
	switch (severity) {
	case PVGuiQt::PVImportPage::Severity::information:
		return QStyle::SP_MessageBoxInformation;
	case PVGuiQt::PVImportPage::Severity::warning:
		return QStyle::SP_MessageBoxWarning;
	case PVGuiQt::PVImportPage::Severity::critical:
		break;
	}
	return QStyle::SP_MessageBoxCritical;
}

/** Takes @p message out of the page at once: deleteLater() would leave it shown until then. */
void remove(QWidget* message)
{
	message->hide();
	message->setParent(nullptr);
	message->deleteLater();
}

} // namespace

PVGuiQt::PVImportPage::PVImportPage(QWidget* parent)
    : QWidget(parent)
    , _title(new QLabel)
    , _messages(new QVBoxLayout)
    , _progress(new QWidget)
    , _task(new QLabel)
    , _bar(new QProgressBar)
    , _details(new QLabel)
    , _progress_buttons(new QHBoxLayout)
    , _close(new QPushButton(tr("Close")))
{
	setObjectName("PVImportPage");
	// As the headers of the start page.
	_title->setFont(QFont("Black Han Sans, Expanded", 20));
	_title->setWordWrap(true);
	_task->setObjectName("import_task");
	_details->setObjectName("import_details");
	_progress->setObjectName("import_progress");
	_close->setObjectName("import_close");
	connect(_close, &QPushButton::clicked, this, &PVImportPage::close_requested);

	auto* progress_layout = new QVBoxLayout(_progress);
	progress_layout->setContentsMargins(0, 0, 0, 0);
	progress_layout->addWidget(_task);
	progress_layout->addWidget(_bar);
	progress_layout->addWidget(_details);
	progress_layout->addLayout(_progress_buttons);
	_progress->hide();

	auto* content = new QWidget;
	auto* content_layout = new QVBoxLayout(content);
	content_layout->addWidget(_title);
	content_layout->addLayout(_messages);
	content_layout->addWidget(_progress);
	content_layout->addStretch();
	auto* scroll_area = new QScrollArea;
	scroll_area->setWidgetResizable(true);
	scroll_area->setFrameShape(QFrame::NoFrame);
	scroll_area->setWidget(content);

	auto* footer = new QHBoxLayout;
	footer->addStretch();
	footer->addWidget(_close);
	_close->hide();

	auto* layout = new QVBoxLayout(this);
	layout->addWidget(scroll_area);
	layout->addLayout(footer);
}

void PVGuiQt::PVImportPage::set_title(QString const& title)
{
	_title->setText(title);
}

void PVGuiQt::PVImportPage::start_progress(QString const& task, std::vector<Action> buttons)
{
	_task->setText(task);
	_bar->setRange(0, 0);
	set_progress_details({});
	while (QLayoutItem* item = _progress_buttons->takeAt(0)) {
		if (item->widget() != nullptr) {
			remove(item->widget());
		}
		delete item;
	}
	_progress_buttons->addStretch();
	for (Action& action : buttons) {
		auto* button = new QPushButton(action.text);
		connect(button, &QPushButton::clicked, this, std::move(action.on_click));
		_progress_buttons->addWidget(button);
		// At once rather than once the event loop shows what a layout adds to a visible widget.
		button->show();
	}
	_progress->show();
}

void PVGuiQt::PVImportPage::set_progress(qint64 value, qint64 maximum)
{
	// Scaled down alike for a progress bar, which counts with ints.
	const qint64 scale = maximum / std::numeric_limits<int>::max() + 1;
	_bar->setRange(0, int(maximum / scale));
	_bar->setValue(int(std::clamp<qint64>(value, 0, maximum) / scale));
}

void PVGuiQt::PVImportPage::set_progress_details(QString const& details)
{
	_details->setText(details);
	_details->setVisible(not details.isEmpty());
}

void PVGuiQt::PVImportPage::set_progress_buttons_enabled(bool enabled)
{
	for (QPushButton* button : _progress->findChildren<QPushButton*>()) {
		button->setEnabled(enabled);
	}
}

void PVGuiQt::PVImportPage::end_progress()
{
	_progress->hide();
}

QWidget* PVGuiQt::PVImportPage::add_message(Severity severity,
                                            QString const& text,
                                            std::vector<Action> actions)
{
	auto* message = new QFrame;
	message->setObjectName("import_message");
	message->setFrameShape(QFrame::StyledPanel);
	auto* icon = new QLabel;
	const int size = style()->pixelMetric(QStyle::PM_MessageBoxIconSize, nullptr, this);
	icon->setPixmap(style()->standardIcon(pixmap_of(severity), nullptr, this).pixmap(size, size));
	auto* label = new QLabel(text);
	label->setWordWrap(true);
	label->setTextInteractionFlags(Qt::TextSelectableByMouse);
	auto* layout = new QHBoxLayout(message);
	layout->addWidget(icon, 0, Qt::AlignTop);
	layout->addWidget(label, 1);
	for (Action& action : actions) {
		auto* button = new QPushButton(action.text);
		layout->addWidget(button, 0, Qt::AlignTop);
		connect(button, &QPushButton::clicked, this,
		        [message, on_click = std::move(action.on_click)]() {
			        remove(message);
			        on_click();
		        });
	}
	_messages->addWidget(message);
	return message;
}

void PVGuiQt::PVImportPage::remove_message(QWidget* message)
{
	_messages->removeWidget(message);
	remove(message);
}

void PVGuiQt::PVImportPage::clear_messages()
{
	while (QLayoutItem* item = _messages->takeAt(0)) {
		remove(item->widget());
		delete item;
	}
}

bool PVGuiQt::PVImportPage::has_messages() const
{
	return _messages->count() > 0;
}

void PVGuiQt::PVImportPage::set_closable(bool closable)
{
	_close->setVisible(closable);
}
