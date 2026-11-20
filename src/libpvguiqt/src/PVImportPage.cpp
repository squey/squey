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

#include <PVFormatBuilderWidget.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QAbstractButton>
#include <QComboBox>
#include <QDropEvent>
#include <QEvent>
#include <QFont>
#include <QFrame>
#include <QHBoxLayout>
#include <QIconEngine>
#include <QKeyEvent>
#include <QLabel>
#include <QLayoutItem>
#include <QMimeData>
#include <QPainter>
#include <QPolygonF>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QStyle>
#include <QThread>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cassert>
#include <exception>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace
{

/** The width of what the page shows at most, in average characters. */
constexpr int COLUMN_WIDTH = 140;
/** Around the sheet, and inside it, in pixels. */
constexpr int SHEET_MARGIN = 24;
/** The radius of the corners of the sheet, as the theme rounds them, in pixels. */
constexpr int SHEET_RADIUS = 8;
/** Between the axes of the drawing behind the sheet, in pixels. */
constexpr int AXIS_SPACING = 160;

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

/** For the theme to tell messages apart. */
const char* name_of(PVGuiQt::PVImportPage::Severity severity)
{
	switch (severity) {
	case PVGuiQt::PVImportPage::Severity::information:
		return "information";
	case PVGuiQt::PVImportPage::Severity::warning:
		return "warning";
	case PVGuiQt::PVImportPage::Severity::critical:
		break;
	}
	return "critical";
}

/** Takes @p widget out of the page at once: deleteLater() would leave it shown until then. */
void remove(QWidget* widget)
{
	widget->hide();
	widget->setParent(nullptr);
	widget->deleteLater();
}

/**
 * The icon of a primary button: light on its fill in either theme, as the theme draws the others
 * once the button is disabled, and so its fill.
 */
class PrimaryIconEngine : public QIconEngine
{
  public:
	explicit PrimaryIconEngine(QString const& name)
	    : _light(":/qss_icons/dark/rc.dark/" + name + "@2x.png"), _themed(PVModdedIcon(name))
	{
	}

	void paint(QPainter* painter, QRect const& rect, QIcon::Mode mode, QIcon::State state) override
	{
		icon(mode).paint(painter, rect, Qt::AlignCenter, mode, state);
	}

	QPixmap pixmap(QSize const& size, QIcon::Mode mode, QIcon::State state) override
	{
		return icon(mode).pixmap(size, mode, state);
	}

	QPixmap
	scaledPixmap(QSize const& size, QIcon::Mode mode, QIcon::State state, qreal scale) override
	{
		return icon(mode).pixmap(size, scale, mode, state);
	}

	QIconEngine* clone() const override { return new PrimaryIconEngine(*this); }

  private:
	QIcon const& icon(QIcon::Mode mode) const { return mode == QIcon::Disabled ? _themed : _light; }

  private:
	QIcon _light;
	QIcon _themed;
};

/** With a space between @p icon and @p text, as the buttons of the start page. */
QPushButton* button(QString const& text, QIcon const& icon)
{
	return new QPushButton(icon, icon.isNull() ? text : " " + text);
}

QPushButton* button(QString const& text, QString const& icon)
{
	return button(text, PVModdedIcon(icon));
}

/** The button of what the page is for, which the theme sets apart. */
QPushButton* primary_button(QString const& text, QString const& icon)
{
	QPushButton* primary = button(text, QIcon(new PrimaryIconEngine(icon)));
	primary->setProperty("primary", true);
	return primary;
}

/**
 * A scroll area as tall as what it shows, for a layout to give it the height it needs, or what
 * there is: that of Qt asks for 24 lines at most.
 */
class FittingScrollArea : public QScrollArea
{
  public:
	QSize sizeHint() const override
	{
		if (widget() == nullptr) {
			return QScrollArea::sizeHint();
		}
		const int frame = 2 * frameWidth();
		return widget()->sizeHint() + QSize(frame, frame);
	}
};

/**
 * Rows made up for the drawing behind the sheet: groups of close rows, with their parallel
 * coordinates, the plot Squey is known for, one of them selected.
 */
struct Rows {
	static constexpr int AXES = 32;

	struct Row {
		std::array<float, AXES> values;
		bool selected;
	};

	Rows()
	{
		// The same drawing in every page.
		std::mt19937 random(5);
		std::uniform_real_distribution<float> uniform(0.08f, 0.92f);
		std::normal_distribution<float> step(0.f, 0.16f);
		constexpr int GROUPS = 6;
		for (int group = 0; group < GROUPS; group++) {
			std::array<float, AXES> profile;
			profile[0] = uniform(random);
			for (int axis = 1; axis < AXES; axis++) {
				profile[axis] = std::clamp(profile[axis - 1] + step(random), 0.05f, 0.95f);
			}
			std::normal_distribution<float> spread(0.f, 0.015f + 0.008f * float(group));
			for (int row = 0; row < 12 + 4 * group; row++) {
				Row& added = rows.emplace_back(Row{.values = {}, .selected = group == 2});
				for (int axis = 0; axis < AXES; axis++) {
					added.values[axis] = std::clamp(profile[axis] + spread(random), 0.f, 1.f);
				}
			}
		}
		for (int row = 0; row < 12; row++) {
			Row& outlier = rows.emplace_back(Row{.values = {}, .selected = false});
			for (float& value : outlier.values) {
				value = uniform(random);
			}
		}
	}

	std::vector<Row> rows;
};

/**
 * The drawing behind the sheet of a page of @p size, for a screen of @p ratio: axes and rows in
 * @p color, selected rows in @p selection_color.
 */
QPixmap decoration(QSize size, qreal ratio, QColor const& color, QColor const& selection_color)
{
	static const Rows made_up;
	QPixmap pixmap(size * ratio);
	pixmap.setDevicePixelRatio(ratio);
	pixmap.fill(Qt::transparent);
	QPainter painter(&pixmap);
	painter.setRenderHint(QPainter::Antialiasing);

	// Axes from one edge to the other, as many on each side of the sheet, at the center.
	const int axes = std::min(Rows::AXES, size.width() / AXIS_SPACING + 2);
	const qreal first = (size.width() - (axes - 1) * AXIS_SPACING) / 2.;
	const qreal top = size.height() * 0.06;
	const qreal height = size.height() * 0.88;
	const auto x_of = [first](int axis) { return first + axis * AXIS_SPACING; };

	QColor axis_color = color;
	axis_color.setAlphaF(0.12f);
	painter.setPen(QPen(axis_color, 1));
	for (int axis = 0; axis < axes; axis++) {
		painter.drawLine(QPointF(x_of(axis), top), QPointF(x_of(axis), top + height));
	}
	// The selection over the other rows, as Squey draws it.
	for (const bool selected : {false, true}) {
		QColor row_color = selected ? selection_color : color;
		row_color.setAlphaF(selected ? 0.28f : 0.045f);
		painter.setPen(QPen(row_color, 1.2));
		for (const Rows::Row& row : made_up.rows) {
			if (row.selected != selected) {
				continue;
			}
			QPolygonF line;
			for (int axis = 0; axis < axes; axis++) {
				line << QPointF(x_of(axis), top + height * (1.f - row.values[axis]));
			}
			painter.drawPolyline(line);
		}
	}
	return pixmap;
}

} // namespace

/** An icon that follows the theme, as PVModdedIcon draws it at each paint. */
class PVGuiQt::PVImportPage::IconLabel : public QWidget
{
  public:
	IconLabel() { setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed); }

	void set_icon(QIcon const& icon)
	{
		_icon = icon;
		setVisible(not icon.isNull());
		update();
	}

	QSize sizeHint() const override
	{
		const int side = 2 * fontMetrics().height();
		return {side, side};
	}

  protected:
	void paintEvent(QPaintEvent*) override
	{
		QPainter painter(this);
		_icon.paint(&painter, rect());
	}

  private:
	QIcon _icon;
};

/** Over the page, while files dragged over it would go to its content: where they go. */
class PVGuiQt::PVImportPage::DropOverlay : public QWidget
{
  public:
	explicit DropOverlay(PVImportPage* page) : QWidget(page), _page(page)
	{
		setObjectName("import_drop_overlay");
		// For the drag to go on to the page under it.
		setAttribute(Qt::WA_TransparentForMouseEvents);
		hide();
	}

	void cover()
	{
		setGeometry(_page->rect());
		raise();
		show();
	}

  protected:
	void paintEvent(QPaintEvent*) override
	{
		QPainter painter(this);
		painter.setRenderHint(QPainter::Antialiasing);
		QColor veil = _page->palette().color(QPalette::Window);
		veil.setAlphaF(0.85f);
		painter.fillRect(rect(), veil);

		const QColor accent = _page->decoration_color();
		QColor fill = accent;
		fill.setAlphaF(0.1f);
		painter.setBrush(fill);
		painter.setPen(QPen(accent, 2, Qt::DashLine));
		const QRectF zone =
		    QRectF(rect()).adjusted(SHEET_MARGIN, SHEET_MARGIN, -SHEET_MARGIN, -SHEET_MARGIN);
		painter.drawRoundedRect(zone, 2 * SHEET_RADIUS, 2 * SHEET_RADIUS);

		const int side = 4 * fontMetrics().height();
		const QPoint center = zone.center().toPoint();
		PVModdedIcon("file-import").paint(&painter, center.x() - side / 2, center.y() - side, side,
		                                  side);
		QFont font = painter.font();
		font.setPointSizeF(font.pointSizeF() * 1.5);
		font.setBold(true);
		painter.setFont(font);
		painter.setPen(_page->palette().color(QPalette::WindowText));
		painter.drawText(QRectF(zone.left(), center.y() + fontMetrics().height(), zone.width(),
		                        zone.bottom() - center.y()),
		                 Qt::AlignHCenter | Qt::AlignTop | Qt::TextWordWrap,
		                 PVImportPage::tr("Drop to add the files to the import"));
	}

  private:
	PVImportPage* _page;
};

PVGuiQt::PVImportPage::PVImportPage(QWidget* parent)
    : QWidget(parent)
    , _layout(new QVBoxLayout(this))
    , _sheet(new QFrame)
    , _back(button(tr("Back"), "arrow-left"))
    , _icon(new IconLabel)
    , _title(new QLabel)
    , _scroll(new FittingScrollArea)
    , _body(new QWidget)
    , _body_layout(new QVBoxLayout(_body))
    , _messages(new QVBoxLayout)
    , _progress(new QWidget)
    , _task(new QLabel)
    , _bar(new QProgressBar)
    , _details(new QLabel)
    , _progress_buttons(new QHBoxLayout)
    , _content_layout(new QVBoxLayout)
    , _collection_row(new QWidget)
    , _collection(new QComboBox)
    , _close(button(tr("Close"), "window-close"))
    , _cancel(button(tr("Cancel"), "circle-x"))
    , _import(primary_button(tr("Import"), "file-import"))
    , _format_cancel(button(tr("Cancel"), "circle-x"))
    , _format_use(primary_button(tr("Use this format"), "circle-check"))
    , _drop_overlay(new DropOverlay(this))
    , _decoration_color(palette().color(QPalette::Highlight))
{
	setObjectName("PVImportPage");
	setAcceptDrops(true);
	_sheet->setObjectName("import_sheet");
	// As the headers of the start page.
	_title->setFont(QFont("Black Han Sans, Expanded", 20));
	_title->setWordWrap(true);
	_icon->hide();
	QFont task_font = _task->font();
	task_font.setBold(true);
	_task->setFont(task_font);
	_back->setObjectName("import_back");
	_task->setObjectName("import_task");
	_details->setObjectName("import_details");
	_progress->setObjectName("import_progress");
	_close->setObjectName("import_close");
	_cancel->setObjectName("import_cancel");
	_import->setObjectName("import_import");
	_import->setToolTip(tr("Import (Enter)"));
	_collection->setObjectName("import_collection");
	_format_cancel->setObjectName("import_format_cancel");
	_format_use->setObjectName("import_format_use");
	connect(_back, &QPushButton::clicked, this, &PVImportPage::back_requested);
	connect(_close, &QPushButton::clicked, this, &PVImportPage::close_requested);
	connect(_cancel, &QPushButton::clicked, this, &PVImportPage::close_requested);
	connect(_import, &QPushButton::clicked, this, &PVImportPage::import_requested);
	connect(_format_cancel, &QPushButton::clicked, this, &PVImportPage::end_format_edit);
	connect(_format_use, &QPushButton::clicked, this, [this]() {
		const QDomDocument format = _builder->get_dom();
		end_format_edit();
		_content->set_edited_format(format);
	});

	_task_timer.setInterval(100);
	connect(&_task_timer, &QTimer::timeout, this, [this]() {
		set_progress(qint64(_task_progress->value()), qint64(_task_progress->maximum()));
		set_progress_details(_task_progress->status());
	});

	auto* collection_layout = new QHBoxLayout(_collection_row);
	collection_layout->setContentsMargins(0, 0, 0, 0);
	collection_layout->addWidget(new QLabel(tr("Data collection:")));
	collection_layout->addWidget(_collection);
	_collection_row->hide();

	auto* progress_layout = new QVBoxLayout(_progress);
	progress_layout->setContentsMargins(0, 0, 0, 0);
	progress_layout->addWidget(_task);
	progress_layout->addWidget(_bar);
	progress_layout->addWidget(_details);
	progress_layout->addLayout(_progress_buttons);
	_progress->hide();

	auto* header = new QFrame;
	header->setObjectName("import_header");
	auto* header_layout = new QHBoxLayout(header);
	header_layout->setContentsMargins(0, 0, 0, SHEET_MARGIN / 2);
	header_layout->setSpacing(SHEET_MARGIN / 2);
	header_layout->addWidget(_back);
	header_layout->addWidget(_icon);
	header_layout->addWidget(_title, 1);

	auto* actions = new QFrame;
	actions->setObjectName("import_actions");
	auto* actions_layout = new QHBoxLayout(actions);
	actions_layout->setContentsMargins(0, SHEET_MARGIN / 2, 0, 0);
	actions_layout->addWidget(_collection_row);
	actions_layout->addStretch();
	actions_layout->addWidget(_close);
	actions_layout->addWidget(_cancel);
	actions_layout->addWidget(_import);
	actions_layout->addWidget(_format_cancel);
	actions_layout->addWidget(_format_use);
	_close->hide();
	_cancel->hide();
	_import->hide();
	_format_cancel->hide();
	_format_use->hide();

	_body_layout->setContentsMargins(0, 0, 0, 0);
	_body_layout->addLayout(_messages);
	_body_layout->addLayout(_content_layout);
	_body_layout->addWidget(_progress);
	_body_layout->addStretch();
	_scroll->setObjectName("import_scroll");
	_scroll->setWidgetResizable(true);
	_scroll->setFrameShape(QFrame::NoFrame);
	_scroll->setWidget(_body);
	// The scroll area keeps to what it shows, which grows or shrinks without telling the layouts
	// around the scroll area.
	_body->installEventFilter(this);

	auto* sheet_layout = new QVBoxLayout(_sheet);
	sheet_layout->setContentsMargins(SHEET_MARGIN, SHEET_MARGIN / 2, SHEET_MARGIN,
	                                 SHEET_MARGIN / 2);
	sheet_layout->addWidget(header);
	sheet_layout->addWidget(_scroll, 1);
	sheet_layout->addWidget(actions);
	// For the shadow the page draws around it.
	_sheet->installEventFilter(this);

	// The sheet in the middle of the room of its row, left to the stretches on each side once
	// as wide as its column: no alignment, which would keep it to the width it asks for.
	auto* row = new QHBoxLayout;
	row->addStretch(1);
	row->addWidget(_sheet, 1000);
	row->addStretch(1);
	_layout->setContentsMargins(SHEET_MARGIN, SHEET_MARGIN, SHEET_MARGIN, SHEET_MARGIN);
	_layout->addLayout(row);
	_layout->addStretch();
	fill_page(false);
}

PVGuiQt::PVImportPage::~PVImportPage()
{
	// The work of a task does not touch the page, but its content may own what it reads.
	if (_task_thread != nullptr) {
		_task_progress->cancel();
		_task_thread->wait();
	}
}

void PVGuiQt::PVImportPage::set_title(QString const& title, QIcon const& icon)
{
	_title->setText(title);
	_icon->set_icon(icon);
}

void PVGuiQt::PVImportPage::set_content(PVWidgets::PVImportWidget* content,
                                        QString const& title,
                                        QIcon const& icon,
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
	connect(content, &PVWidgets::PVImportWidget::format_edit_requested, this,
	        &PVImportPage::edit_format);

	_collection->clear();
	_collection->addItems(collections);
	_collection->addItem(tr("New data collection"));
	_collection->setCurrentIndex(current < 0 ? int(collections.size()) : current);

	set_title(title, icon);
	set_closable(false);
	fill_page(content->fills_page());
	set_content_shown(true);
}

void PVGuiQt::PVImportPage::clear_content()
{
	if (_content == nullptr) {
		return;
	}
	assert(not task_running());
	if (editing_format()) {
		end_format_edit();
	}
	remove(_content);
	_content = nullptr;
	fill_page(false);
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
	// The format builder stands for the content while the user makes its format.
	const bool content_shown = shown and not editing_format();
	if (_content != nullptr) {
		_content->setVisible(content_shown);
	}
	if (editing_format()) {
		_builder->setVisible(shown);
	}
	_collection_row->setVisible(content_shown);
	_cancel->setVisible(content_shown);
	_import->setVisible(content_shown);
	_format_cancel->setVisible(shown and editing_format());
	_format_use->setVisible(shown and editing_format());
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

void PVGuiQt::PVImportPage::edit_format(QDomDocument const& format,
                                        PVRush::PVInputDescription_p const& input,
                                        std::shared_ptr<PVRush::PVInputType> const& type)
{
	assert(_content != nullptr and not editing_format());
	_builder = new App::PVFormatBuilderWidget;
	_builder->setWindowFlags(Qt::Widget);
	if (not format.documentElement().isNull()) {
		QDomDocument document = format;
		_builder->openFormat(document);
	}
	_content_layout->addWidget(_builder);
	set_content_shown(true);
	fill_page(true);
	_builder->preview_input(input, type);
}

void PVGuiQt::PVImportPage::end_format_edit()
{
	remove(_builder);
	_builder = nullptr;
	fill_page(_content->fills_page());
	set_content_shown(true);
}

void PVGuiQt::PVImportPage::fill_page(bool fill)
{
	// Else short enough lines to read in a wide window, and a sheet no taller than what it shows,
	// with its buttons right under it.
	_sheet->setMaximumWidth(fill ? QWIDGETSIZE_MAX
	                             : fontMetrics().averageCharWidth() * COLUMN_WIDTH +
	                                   2 * SHEET_MARGIN);
	_sheet->setSizePolicy(QSizePolicy::Expanding,
	                      fill ? QSizePolicy::Expanding : QSizePolicy::Maximum);
	_layout->setStretch(0, fill ? 1 : 0);
	_layout->setStretch(1, fill ? 0 : 1);
	// Else the height left goes below the content: given to the content, it would set a list of
	// a limited height apart from the buttons beside it.
	_body_layout->setStretchFactor(_content_layout, fill ? 1 : 0);
	_body_layout->setStretch(_body_layout->count() - 1, fill ? 0 : 1);
}

void PVGuiQt::PVImportPage::update_buttons()
{
	_import->setEnabled(_content != nullptr and _content->is_ready() and not task_running());
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
		QPushButton* button = ::button(action.text, action.icon);
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
	message->setProperty("severity", name_of(severity));
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
		QPushButton* button = ::button(action.text, action.icon);
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

void PVGuiQt::PVImportPage::set_back_enabled(bool enabled)
{
	_back->setEnabled(enabled);
}

void PVGuiQt::PVImportPage::cancel_task()
{
	if (_task_progress) {
		_task_progress->cancel();
	}
}

void PVGuiQt::PVImportPage::set_decoration_color(QColor const& color)
{
	_decoration_color = color;
	_decoration = {};
	update();
}

void PVGuiQt::PVImportPage::paintEvent(QPaintEvent*)
{
	QPainter painter(this);
	painter.fillRect(rect(), palette().color(QPalette::Window));
	const qreal ratio = devicePixelRatioF();
	if (_decoration.size() != size() * ratio) {
		_decoration =
		    decoration(size(), ratio, palette().color(QPalette::WindowText), _decoration_color);
	}
	painter.drawPixmap(0, 0, _decoration);

	// A shadow, wider and fainter from the sheet, a bit below it.
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setPen(Qt::NoPen);
	painter.setBrush(QColor(0, 0, 0, 7));
	const QRectF sheet = _sheet->geometry();
	for (int spread = 1; spread <= 8; spread++) {
		painter.drawRoundedRect(sheet.adjusted(-spread, 2 - spread, spread, 2 + spread),
		                        SHEET_RADIUS + spread, SHEET_RADIUS + spread);
	}
}

void PVGuiQt::PVImportPage::changeEvent(QEvent* event)
{
	// The colors of a theme, which polishes the page anew.
	if (event->type() == QEvent::PaletteChange or event->type() == QEvent::StyleChange) {
		_decoration = {};
		update();
	}
	QWidget::changeEvent(event);
}

void PVGuiQt::PVImportPage::keyPressEvent(QKeyEvent* event)
{
	// As a dialog does with its default button, but on a button, which Enter would not click.
	const bool enter = (event->key() == Qt::Key_Return or event->key() == Qt::Key_Enter) and
	                   (event->modifiers() & ~Qt::KeypadModifier) == Qt::NoModifier;
	if (enter and _import->isVisible() and _import->isEnabled() and
	    qobject_cast<QAbstractButton*>(focusWidget()) == nullptr) {
		_import->animateClick();
		return;
	}
	QWidget::keyPressEvent(event);
}

bool PVGuiQt::PVImportPage::eventFilter(QObject* watched, QEvent* event)
{
	if (watched == _body and event->type() == QEvent::LayoutRequest) {
		_scroll->updateGeometry();
	} else if (watched == _sheet and
	           (event->type() == QEvent::Move or event->type() == QEvent::Resize)) {
		update();
	}
	return QWidget::eventFilter(watched, event);
}

QStringList PVGuiQt::PVImportPage::dropped_files(QMimeData const* mime)
{
	QStringList paths;
	for (const QUrl& url : mime->urls()) {
		if (not url.isLocalFile()) {
			return {};
		}
		paths << url.toLocalFile();
	}
	return paths;
}

bool PVGuiQt::PVImportPage::takes_dropped_files() const
{
	return _content != nullptr and _content->isVisible() and _content->isEnabled() and
	       _content->takes_files();
}

void PVGuiQt::PVImportPage::dragEnterEvent(QDragEnterEvent* event)
{
	// Else ignored, for the main window to tell what it makes of them.
	if (takes_dropped_files() and not dropped_files(event->mimeData()).isEmpty()) {
		event->acceptProposedAction();
		_drop_overlay->cover();
	}
}

void PVGuiQt::PVImportPage::dragLeaveEvent(QDragLeaveEvent*)
{
	_drop_overlay->hide();
}

void PVGuiQt::PVImportPage::dropEvent(QDropEvent* event)
{
	_drop_overlay->hide();
	const QStringList paths = dropped_files(event->mimeData());
	if (takes_dropped_files() and not paths.isEmpty()) {
		event->acceptProposedAction();
		_content->add_paths(paths);
	}
}
