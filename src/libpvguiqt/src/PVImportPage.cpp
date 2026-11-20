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

#include <pvkernel/core/PVTaskProgress.h>

#include <QComboBox>
#include <QFont>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLayoutItem>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QStyle>
#include <QThread>
#include <QVBoxLayout>

#include <algorithm>
#include <cassert>
#include <exception>
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

/** Takes @p widget out of the page at once: deleteLater() would leave it shown until then. */
void remove(QWidget* widget)
{
	widget->hide();
	widget->setParent(nullptr);
	widget->deleteLater();
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
    , _content_layout(new QVBoxLayout)
    , _collection_row(new QWidget)
    , _collection(new QComboBox)
    , _close(new QPushButton(tr("Close")))
    , _cancel(new QPushButton(tr("Cancel")))
    , _import(new QPushButton(tr("Import")))
{
	setObjectName("PVImportPage");
	// As the headers of the start page.
	_title->setFont(QFont("Black Han Sans, Expanded", 20));
	_title->setWordWrap(true);
	_task->setObjectName("import_task");
	_details->setObjectName("import_details");
	_progress->setObjectName("import_progress");
	_close->setObjectName("import_close");
	_cancel->setObjectName("import_cancel");
	_import->setObjectName("import_import");
	_collection->setObjectName("import_collection");
	connect(_close, &QPushButton::clicked, this, &PVImportPage::close_requested);
	connect(_cancel, &QPushButton::clicked, this, &PVImportPage::cancel_requested);
	connect(_import, &QPushButton::clicked, this, &PVImportPage::import_requested);

	_task_timer.setInterval(100);
	connect(&_task_timer, &QTimer::timeout, this, [this]() {
		set_progress(qint64(_task_progress->value()), qint64(_task_progress->maximum()));
		set_progress_details(_task_progress->status());
	});

	auto* collection_layout = new QHBoxLayout(_collection_row);
	collection_layout->setContentsMargins(0, 0, 0, 0);
	collection_layout->addWidget(new QLabel(tr("Data collection:")));
	collection_layout->addWidget(_collection);
	collection_layout->addStretch();
	_collection_row->hide();

	auto* progress_layout = new QVBoxLayout(_progress);
	progress_layout->setContentsMargins(0, 0, 0, 0);
	progress_layout->addWidget(_task);
	progress_layout->addWidget(_bar);
	progress_layout->addWidget(_details);
	progress_layout->addLayout(_progress_buttons);
	_progress->hide();

	auto* body = new QWidget;
	auto* body_layout = new QVBoxLayout(body);
	body_layout->addWidget(_title);
	body_layout->addLayout(_messages);
	body_layout->addLayout(_content_layout);
	body_layout->addWidget(_collection_row);
	body_layout->addWidget(_progress);
	body_layout->addStretch();
	auto* scroll_area = new QScrollArea;
	scroll_area->setWidgetResizable(true);
	scroll_area->setFrameShape(QFrame::NoFrame);
	scroll_area->setWidget(body);

	auto* footer = new QHBoxLayout;
	footer->addStretch();
	footer->addWidget(_close);
	footer->addWidget(_cancel);
	footer->addWidget(_import);
	_close->hide();
	_cancel->hide();
	_import->hide();

	auto* layout = new QVBoxLayout(this);
	layout->addWidget(scroll_area);
	layout->addLayout(footer);
}

PVGuiQt::PVImportPage::~PVImportPage()
{
	// The work of a task does not touch the page, but its content may own what it reads.
	if (_task_thread != nullptr) {
		_task_progress->cancel();
		_task_thread->wait();
	}
}

void PVGuiQt::PVImportPage::set_title(QString const& title)
{
	_title->setText(title);
}

void PVGuiQt::PVImportPage::set_content(PVWidgets::PVImportWidget* content,
                                        QString const& title,
                                        QStringList const& collections,
                                        int current)
{
	clear_content();
	_content = content;
	_content_layout->addWidget(content);
	content->set_task_runner([this](QString const& task, auto work, auto done) {
		run_task(task, std::move(work), std::move(done));
	});
	connect(content, &PVWidgets::PVImportWidget::ready_changed, this,
	        &PVImportPage::update_buttons);
	connect(content, &PVWidgets::PVImportWidget::message, this,
	        [this](Severity severity, QString const& text) { add_message(severity, text); });

	_collection->clear();
	_collection->addItems(collections);
	_collection->addItem(tr("New data collection"));
	_collection->setCurrentIndex(current < 0 ? int(collections.size()) : current);

	set_title(title);
	set_closable(false);
	set_content_shown(true);
}

void PVGuiQt::PVImportPage::clear_content()
{
	if (_content == nullptr) {
		return;
	}
	assert(not task_running());
	remove(_content);
	_content = nullptr;
	set_content_shown(false);
}

int PVGuiQt::PVImportPage::chosen_collection() const
{
	const int index = _collection->currentIndex();
	return index == _collection->count() - 1 ? -1 : index;
}

void PVGuiQt::PVImportPage::set_content_shown(bool shown)
{
	shown = shown and _content != nullptr;
	if (_content != nullptr) {
		_content->setVisible(shown);
	}
	_collection_row->setVisible(shown);
	_cancel->setVisible(shown);
	_import->setVisible(shown);
	update_buttons();
}

void PVGuiQt::PVImportPage::run_task(QString const& title,
                                     std::function<void(PVCore::PVTaskProgress&)> work,
                                     std::function<void(bool completed)> done)
{
	assert(not task_running());
	auto progress = std::make_shared<PVCore::PVTaskProgress>();
	auto error = std::make_shared<QString>();
	_task_progress = progress;
	start_progress(title, {{tr("Cancel"), [progress]() { progress->cancel(); }}});
	_content->setEnabled(false);

	_task_thread = QThread::create([work = std::move(work), progress, error]() {
		try {
			work(*progress);
		} catch (const std::exception& e) {
			*error = QString::fromStdString(e.what());
		}
	});
	connect(_task_thread, &QThread::finished, this,
	        [this, progress, error, done = std::move(done)]() {
		        _task_timer.stop();
		        _task_thread->deleteLater();
		        _task_thread = nullptr;
		        _task_progress.reset();
		        end_progress();
		        _content->setEnabled(true);
		        update_buttons();
		        if (not error->isEmpty()) {
			        add_message(Severity::critical, error->toHtmlEscaped());
		        }
		        done(error->isEmpty() and not progress->canceled());
		        Q_EMIT task_finished();
	        });
	update_buttons();
	_task_timer.start();
	_task_thread->start();
}

void PVGuiQt::PVImportPage::update_buttons()
{
	_import->setEnabled(_content != nullptr and _content->is_ready() and not task_running());
	// Its task would outlive the content.
	_cancel->setEnabled(not task_running());
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
