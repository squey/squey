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
//

#include <pvguiqt/PVSettingsDialog.h>

#include <pvguiqt/PVCodeEditorStyle.h>
#include <pvguiqt/PVNrawDirectoryMessageBox.h>

#include <pvkernel/core/PVFlatpakHelper.h>
#include <pvkernel/core/PVKernelSettings.h>
#include <pvkernel/core/PVSettings.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QIconEngine>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QScrollBar>
#include <QShortcut>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QStyle>
#include <QTextBlock>
#include <QTextEdit>
#include <QTimer>
#include <QToolButton>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <functional>

namespace
{

constexpr int SECTION_KEY_ROLE = Qt::UserRole;
constexpr int LINE_ROLE = Qt::UserRole;

/** The pages of the dialog, in its stacked widget. */
constexpr int FORM_PAGE = 0;
constexpr int JSON_PAGE = 1;

QString section_of(QString const& key)
{
	return key.section('.', 0, -2);
}

/** How settings.json writes \p value. */
QString json_text(QJsonValue const& value)
{
	// A document holds an array or an object, never a bare value.
	const QByteArray array = QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact);
	return QString::fromUtf8(array.mid(1, array.size() - 2));
}

QString document_text(QJsonObject const& settings)
{
	return QString::fromUtf8(QJsonDocument(settings).toJson(QJsonDocument::Indented));
}

/** The words of \p text, in lines of at most \p width characters when a word allows it. */
QStringList wrapped(QString const& text, int width)
{
	QStringList lines;
	QString line;
	for (QString const& word : text.split(' ', Qt::SkipEmptyParts)) {
		if (not line.isEmpty() and line.size() + 1 + word.size() > width) {
			lines << line;
			line.clear();
		}
		line += (line.isEmpty() ? "" : " ") + word;
	}
	if (not line.isEmpty()) {
		lines << line;
	}
	return lines;
}

/** What a setting accepts, as a sentence, or nothing when its type says it all. */
QString accepted_values(PVCore::PVSettingDescriptor const& descriptor)
{
	switch (descriptor.type) {
	case PVCore::PVSettingType::Choice: {
		QStringList choices;
		for (PVCore::PVSettingChoice const& choice : descriptor.choices) {
			choices << QString("%1 (%2)").arg(json_text(choice.value), choice.label);
		}
		if (descriptor.machine_choices) {
			return QString("One of %1, or one of the choices the form lists for this machine.")
			    .arg(choices.join(", "));
		}
		return QString("One of %1.").arg(choices.join(", "));
	}
	case PVCore::PVSettingType::Int:
	case PVCore::PVSettingType::Double: {
		const QString unit = descriptor.unit.isEmpty() ? QString() : " " + descriptor.unit;
		const QString minimum = json_text(QJsonValue::fromVariant(descriptor.minimum));
		const QString maximum = json_text(QJsonValue::fromVariant(descriptor.maximum));
		if (descriptor.minimum.isValid() and descriptor.maximum.isValid()) {
			return QString("From %1 to %2%3.").arg(minimum, maximum, unit);
		}
		if (descriptor.minimum.isValid()) {
			return QString("%1%2 or more.").arg(minimum, unit);
		}
		if (descriptor.maximum.isValid()) {
			return QString("%1%2 or less.").arg(maximum, unit);
		}
		return descriptor.unit.isEmpty() ? QString() : QString("In %1.").arg(descriptor.unit);
	}
	default:
		return {};
	}
}

/** The line of \p text where settings.json sets \p key, counted from 1, or 0. */
int line_of_key(QString const& text, QString const& key)
{
	const QRegularExpression expression(
	    QString("\"%1\"\\s*:").arg(QRegularExpression::escape(key)));
	const QRegularExpressionMatch match = expression.match(text);
	return match.hasMatch() ? text.left(match.capturedStart()).count('\n') + 1 : 0;
}

/**
 * Leaves the wheel to the list of settings over an editor that does not have
 * the focus: scrolling through the list would otherwise change every value it
 * passes over.
 */
class wheel_guard : public QObject
{
  public:
	explicit wheel_guard(QScrollArea* area) : QObject(area), _area(area) {}

	bool eventFilter(QObject* watched, QEvent* event) override
	{
		auto* widget = qobject_cast<QWidget*>(watched);
		if (event->type() == QEvent::Wheel and widget and not widget->hasFocus()) {
			QCoreApplication::sendEvent(_area->verticalScrollBar(), event);
			return true;
		}
		return false;
	}

  private:
	QScrollArea* _area;
};

#ifdef __linux__
/** What a status icon shows. */
enum class status_icon {
	/** A green disc with a check mark. */
	check,
	/** An orange triangle with an exclamation mark. */
	warning,
	/** A red disc with a cross. */
	cross
};

/** A status icon, drawn at the size and scale it is asked for. */
class status_icon_engine : public QIconEngine
{
  public:
	explicit status_icon_engine(status_icon icon) : _icon(icon) {}

	QIconEngine* clone() const override { return new status_icon_engine(*this); }

	void paint(QPainter* painter, QRect const& rect, QIcon::Mode mode, QIcon::State state) override
	{
		painter->drawPixmap(
		    rect, scaledPixmap(rect.size(), mode, state, painter->device()->devicePixelRatioF()));
	}

	QPixmap pixmap(QSize const& size, QIcon::Mode mode, QIcon::State state) override
	{
		return scaledPixmap(size, mode, state, 1);
	}

	QPixmap scaledPixmap(QSize const& size, QIcon::Mode mode, QIcon::State, qreal scale) override
	{
		QPixmap pixmap(size * scale);
		pixmap.setDevicePixelRatio(scale);
		pixmap.fill(Qt::transparent);
		QPainter painter(&pixmap);
		painter.setRenderHint(QPainter::Antialiasing);
		if (mode == QIcon::Disabled) {
			painter.setOpacity(0.4);
		}
		// On a grid of 16 by 16, in the middle of the pixmap.
		const qreal side = std::min(size.width(), size.height());
		painter.translate((size.width() - side) / 2, (size.height() - side) / 2);
		painter.scale(side / 16, side / 16);

		switch (_icon) {
		case status_icon::check:
			painter.setPen(Qt::NoPen);
			painter.setBrush(QColor(0x2e, 0xa0, 0x43));
			painter.drawEllipse(QRectF(0.5, 0.5, 15, 15));
			painter.setPen(QPen(Qt::white, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
			painter.setBrush(Qt::NoBrush);
			painter.drawPolyline(QPolygonF{{4.5, 8.3}, {7, 10.8}, {11.5, 5.6}});
			break;
		case status_icon::warning: {
			const QColor orange(0xf0, 0x88, 0x00);
			QPainterPath triangle;
			triangle.moveTo(8, 1.5);
			triangle.lineTo(15, 14.5);
			triangle.lineTo(1, 14.5);
			triangle.closeSubpath();
			// The width of the pen rounds the corners.
			painter.setPen(QPen(orange, 1.5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
			painter.setBrush(orange);
			painter.drawPath(triangle);
			painter.setPen(Qt::NoPen);
			painter.setBrush(QColor(0x20, 0x20, 0x20));
			painter.drawRoundedRect(QRectF(7.15, 5.2, 1.7, 5.3), 0.85, 0.85);
			painter.drawEllipse(QRectF(7.15, 11.4, 1.7, 1.7));
			break;
		}
		case status_icon::cross:
			painter.setPen(Qt::NoPen);
			painter.setBrush(QColor(0xda, 0x36, 0x33));
			painter.drawEllipse(QRectF(0.5, 0.5, 15, 15));
			painter.setPen(QPen(Qt::white, 2, Qt::SolidLine, Qt::RoundCap));
			painter.drawLine(QPointF(5.2, 5.2), QPointF(10.8, 10.8));
			painter.drawLine(QPointF(10.8, 5.2), QPointF(5.2, 10.8));
			break;
		}
		return pixmap;
	}

  private:
	status_icon _icon;
};

/**
 * Beside the working directory, how Squey reaches it: a check mark when it writes there
 * directly, a warning when it writes there through the portal of the desktop, slowly, and a
 * cross when it cannot write there. A click explains a directory the Flatpak reaches through
 * the portal or does not see, and gives the command that lets it reach the directory.
 */
class directory_status : public QToolButton
{
  public:
	explicit directory_status(QLineEdit* line_edit) : _line_edit(line_edit)
	{
		setObjectName("status");
		setAccessibleName(tr("Status of the working directory"));
		setAutoRaise(true);
		// Checking creates a directory in the one typed: it waits for the typing to stop.
		_typing.setSingleShot(true);
		_typing.setInterval(400);
		connect(&_typing, &QTimer::timeout, this, [this]() { show_status(); });
		connect(line_edit, &QLineEdit::textChanged, this, [this]() {
			if (_line_edit->hasFocus()) {
				_typing.start();
			} else {
				show_status();
			}
		});
		connect(this, &QToolButton::clicked, this,
		        [this]() { PVNrawDirectoryMessageBox::explain(this, _directory, _reach); });
	}

  private:
	void show_status()
	{
		using reach = PVNrawDirectoryMessageBox::reach;
		_typing.stop();
		_directory = _line_edit->text();
		_reach = PVNrawDirectoryMessageBox::reach_of(_directory);
		status_icon icon = status_icon::cross;
		switch (_reach) {
		case reach::direct:
			icon = status_icon::check;
			setToolTip(tr("Squey writes in this directory directly."));
			break;
		case reach::through_portal:
			icon = status_icon::warning;
			setToolTip(tr("Squey reaches this directory through the portal of the desktop, "
			              "slowly. Click to see how to let it reach the directory directly."));
			break;
		case reach::out_of_sight:
			setToolTip(tr("The Flatpak of Squey does not see this directory. Click to see how "
			              "to give it."));
			break;
		case reach::out_of_reach:
			setToolTip(tr("Squey cannot write in this directory."));
			break;
		}
		setIcon(QIcon(new status_icon_engine(icon)));
		const bool explained = _reach == reach::through_portal or _reach == reach::out_of_sight;
		setCursor(explained ? Qt::PointingHandCursor : Qt::ArrowCursor);
	}

  private:
	QLineEdit* _line_edit;
	QTimer _typing;
	QString _directory;
	PVNrawDirectoryMessageBox::reach _reach = PVNrawDirectoryMessageBox::reach::out_of_reach;
};
#endif

} // namespace

namespace PVGuiQt::__impl
{

/**
 * One setting of the form: its title, what it is for, and an editor of its value.
 */
class setting_row : public QFrame
{
  public:
	setting_row(PVCore::PVSettingDescriptor descriptor, wheel_guard* guard, QWidget* parent);

	PVCore::PVSettingDescriptor const& descriptor() const { return _descriptor; }
	QString section() const { return section_of(_descriptor.key); }
	bool is_modified() const { return _modified; }

	/**
	 * Shows \p value, which the draft sets when \p modified. \p problem says why
	 * the value of the draft is ignored, when it is.
	 */
	void show_value(QVariant const& value, bool modified, bool locked, QString const& problem);

	/** Every word is in the texts of the setting, or in \p section_title. */
	bool matches(QStringList const& words, QString const& section_title) const;

  public:
	std::function<void(QVariant const&)> edited;
	std::function<void()> reset;

  private:
	QWidget* create_editor();
	void set_editor_value(QVariant const& value);
	void notify(QVariant const& value);
	void show_menu(QPoint const& position);

  private:
	PVCore::PVSettingDescriptor _descriptor;
	/** Those of a Choice setting the editor lists. */
	std::vector<PVCore::PVSettingChoice> _choices;
	/** Every text the search matches, but the title of the section. */
	QString _haystack;
	QVariant _value;
	bool _modified = false;
	/** show_value() is setting the editor, which is no edit of the user. */
	bool _showing = false;

	QFrame* _bar;
	QWidget* _editor;
	QToolButton* _reset_button;
	QLabel* _notes;
};

} // namespace PVGuiQt::__impl

PVGuiQt::__impl::setting_row::setting_row(PVCore::PVSettingDescriptor descriptor,
                                          wheel_guard* guard,
                                          QWidget* parent)
    : QFrame(parent)
    , _descriptor(std::move(descriptor))
    , _choices(_descriptor.machine_choices ? _descriptor.machine_choices() : _descriptor.choices)
{
	setObjectName("setting:" + _descriptor.key);
	setContextMenuPolicy(Qt::CustomContextMenu);
	connect(this, &QWidget::customContextMenuRequested, this,
	        [this](QPoint const& position) { show_menu(position); });

	QStringList texts{_descriptor.key, _descriptor.title, _descriptor.description,
	                  _descriptor.unit};
	texts << _descriptor.keywords;
	for (PVCore::PVSettingChoice const& choice : _choices) {
		texts << choice.value << choice.label;
	}
	_haystack = texts.join('\n');

	_bar = new QFrame;
	_bar->setFixedWidth(3);

	auto* title = new QLabel(QString("<b>%1</b>").arg(_descriptor.title.toHtmlEscaped()));
	auto* key = new QLabel(_descriptor.key);
	// Dimmed, as a detail for whoever edits settings.json.
	key->setEnabled(false);

	_reset_button = new QToolButton;
	_reset_button->setObjectName("reset");
	_reset_button->setIcon(PVModdedIcon("arrow-rotate-left"));
	_reset_button->setToolTip(tr("Reset to the default value"));
	_reset_button->setAutoRaise(true);
	connect(_reset_button, &QToolButton::clicked, this, [this]() { reset(); });

	auto* description = new QLabel(_descriptor.description);
	description->setWordWrap(true);
	description->setVisible(not _descriptor.description.isEmpty());

	_editor = create_editor();
	_editor->setObjectName("editor:" + _descriptor.key);
	_editor->installEventFilter(guard);

	_notes = new QLabel;
	_notes->setObjectName("notes");
	_notes->setWordWrap(true);
	QFont notes_font = _notes->font();
	notes_font.setItalic(true);
	_notes->setFont(notes_font);
	_notes->hide();

	auto* header = new QHBoxLayout;
	header->addWidget(title);
	header->addWidget(key);
	header->addStretch();
	header->addWidget(_reset_button);

	auto* content = new QVBoxLayout;
	content->addLayout(header);
	if (_descriptor.type == PVCore::PVSettingType::Bool) {
		auto* line = new QHBoxLayout;
		line->addWidget(_editor, 0, Qt::AlignTop);
		line->addWidget(description, 1);
		content->addLayout(line);
	} else {
		content->addWidget(description);
		const bool text = _descriptor.type == PVCore::PVSettingType::String or
		                  _descriptor.type == PVCore::PVSettingType::Path;
		content->addWidget(_editor, 0, text ? Qt::Alignment() : Qt::AlignLeft);
	}
	content->addWidget(_notes);

	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 6, 6, 6);
	layout->addWidget(_bar);
	layout->addLayout(content, 1);
}

QWidget* PVGuiQt::__impl::setting_row::create_editor()
{
	auto bound = [](QVariant const& limit, auto unset) {
		return limit.isValid() ? limit.value<decltype(unset)>() : unset;
	};
	const QString suffix = _descriptor.unit.isEmpty() ? QString() : " " + _descriptor.unit;

	switch (_descriptor.type) {
	case PVCore::PVSettingType::Bool: {
		auto* check_box = new QCheckBox;
		connect(check_box, &QCheckBox::toggled, this, [this](bool checked) { notify(checked); });
		return check_box;
	}
	case PVCore::PVSettingType::Int: {
		auto* spin_box = new QSpinBox;
		spin_box->setRange(bound(_descriptor.minimum, INT_MIN),
		                   bound(_descriptor.maximum, INT_MAX));
		spin_box->setSuffix(suffix);
		spin_box->setFocusPolicy(Qt::StrongFocus);
		connect(spin_box, &QSpinBox::valueChanged, this,
		        [this](int value) { notify(qlonglong(value)); });
		return spin_box;
	}
	case PVCore::PVSettingType::Double: {
		auto* spin_box = new QDoubleSpinBox;
		spin_box->setDecimals(3);
		spin_box->setRange(bound(_descriptor.minimum, -DBL_MAX),
		                   bound(_descriptor.maximum, DBL_MAX));
		spin_box->setSuffix(suffix);
		spin_box->setFocusPolicy(Qt::StrongFocus);
		connect(spin_box, &QDoubleSpinBox::valueChanged, this,
		        [this](double value) { notify(value); });
		return spin_box;
	}
	case PVCore::PVSettingType::String: {
		auto* line_edit = new QLineEdit;
		if (_descriptor.max_length >= 0) {
			line_edit->setMaxLength(_descriptor.max_length);
		}
		connect(line_edit, &QLineEdit::textEdited, this,
		        [this](QString const& text) { notify(text); });
		return line_edit;
	}
	case PVCore::PVSettingType::Path: {
		auto* path = new QWidget;
		auto* line_edit = new QLineEdit;
		auto* browse = new QToolButton;
		browse->setText("...");
		browse->setToolTip(tr("Choose a directory"));
		auto* layout = new QHBoxLayout(path);
		layout->setContentsMargins(0, 0, 0, 0);
		layout->addWidget(line_edit, 1);
		layout->addWidget(browse);
#ifdef __linux__
		if (_descriptor.key == PVCore::settings::working_directory.key()) {
			layout->addWidget(new directory_status(line_edit));
		}
#endif
		connect(line_edit, &QLineEdit::textEdited, this,
		        [this](QString const& text) { notify(text); });
		connect(browse, &QToolButton::clicked, this, [this, line_edit]() {
			QString directory = PVWidgets::PVFileDialog::getExistingDirectory(
			    this, _descriptor.title, PVCore::PVFlatpakHelper::host_path(line_edit->text()));
			if (not directory.isEmpty()) {
				if (_descriptor.key == PVCore::settings::working_directory.key()) {
					directory = PVNrawDirectoryMessageBox::working_directory_for(this, directory);
				}
				line_edit->setText(directory);
				notify(directory);
			}
		});
		return path;
	}
	case PVCore::PVSettingType::Choice: {
		auto* combo_box = new QComboBox;
		for (PVCore::PVSettingChoice const& choice : _choices) {
			combo_box->addItem(choice.label, choice.value);
		}
		combo_box->setFocusPolicy(Qt::StrongFocus);
		connect(combo_box, &QComboBox::currentIndexChanged, this, [this, combo_box](int index) {
			if (index >= 0) {
				notify(combo_box->itemData(index));
			}
		});
		return combo_box;
	}
	}
	return new QWidget;
}

void PVGuiQt::__impl::setting_row::show_value(QVariant const& value,
                                              bool modified,
                                              bool locked,
                                              QString const& problem)
{
	_value = value;
	_showing = true;
	set_editor_value(value);
	_showing = false;

	_modified = modified;
	_bar->setStyleSheet(modified ? "background-color: palette(highlight);"
	                             : "background-color: transparent;");
	_reset_button->setVisible(modified);
	_editor->setEnabled(not locked);

	QStringList notes;
	if (locked) {
		notes << tr("Set by the administrator.");
	}
	if (not problem.isEmpty()) {
		notes << problem;
	}
	if (_descriptor.requires_restart) {
		notes << tr("Applies after a restart.");
	}
	_notes->setText(notes.join(' '));
	_notes->setVisible(not notes.isEmpty());
}

bool PVGuiQt::__impl::setting_row::matches(QStringList const& words,
                                           QString const& section_title) const
{
	for (QString const& word : words) {
		if (not _haystack.contains(word, Qt::CaseInsensitive) and
		    not section_title.contains(word, Qt::CaseInsensitive)) {
			return false;
		}
	}
	return true;
}

void PVGuiQt::__impl::setting_row::set_editor_value(QVariant const& value)
{
	// The value of the editor the user is typing in is left alone: setting it
	// again would rewrite the text under the cursor.
	if (auto* check_box = qobject_cast<QCheckBox*>(_editor)) {
		check_box->setChecked(value.toBool());
	} else if (auto* spin_box = qobject_cast<QSpinBox*>(_editor)) {
		if (spin_box->value() != int(value.toLongLong())) {
			spin_box->setValue(int(value.toLongLong()));
		}
	} else if (auto* spin_box = qobject_cast<QDoubleSpinBox*>(_editor)) {
		if (spin_box->value() != value.toDouble()) {
			spin_box->setValue(value.toDouble());
		}
	} else if (auto* combo_box = qobject_cast<QComboBox*>(_editor)) {
		int index = combo_box->findData(value.toString());
		if (index < 0 and _descriptor.machine_choices) {
			// A value of another machine, or one that this machine no longer offers.
			const auto known = std::ranges::find(_descriptor.choices, value.toString(),
			                                     &PVCore::PVSettingChoice::value);
			combo_box->addItem(known != _descriptor.choices.end()
			                       ? known->label
			                       : tr("%1, not found on this machine").arg(value.toString()),
			                   value.toString());
			index = combo_box->count() - 1;
		}
		combo_box->setCurrentIndex(index);
	} else {
		// A string, or a path, whose line edit is beside a button.
		auto* line_edit = qobject_cast<QLineEdit*>(_editor);
		if (not line_edit) {
			line_edit = _editor->findChild<QLineEdit*>();
		}
		if (line_edit and line_edit->text() != value.toString()) {
			line_edit->setText(value.toString());
		}
	}
}

void PVGuiQt::__impl::setting_row::notify(QVariant const& value)
{
	if (not _showing) {
		edited(value);
	}
}

void PVGuiQt::__impl::setting_row::show_menu(QPoint const& position)
{
	QMenu menu;
	QAction* reset_action =
	    menu.addAction(PVModdedIcon("arrow-rotate-left"), tr("Reset to the default value"));
	reset_action->setEnabled(_modified);
	QAction* copy_key = menu.addAction(PVModdedIcon("copy"), tr("Copy the key"));
	QAction* copy_json = menu.addAction(tr("Copy as JSON"));

	QAction* chosen = menu.exec(mapToGlobal(position));
	if (chosen == reset_action) {
		reset();
	} else if (chosen == copy_key) {
		QApplication::clipboard()->setText(_descriptor.key);
	} else if (chosen == copy_json) {
		QApplication::clipboard()->setText(
		    QString("\"%1\": %2")
		        .arg(_descriptor.key, json_text(PVCore::PVSettings::to_json(_descriptor, _value))));
	}
}

PVGuiQt::PVSettingsDialog::PVSettingsDialog(QWidget* parent) : QDialog(parent)
{
	setWindowTitle(tr("Settings"));
	resize(1100, 760);

	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	// settings.json may have been edited outside since it was read.
	settings.reload();
	_draft = settings.user_settings();

	_search = new QLineEdit;
	_search->setObjectName("search");
	_search->setPlaceholderText(
	    tr("Search the settings by name or description, @modified for those changed"));
	_search->setClearButtonEnabled(true);
	_search->addAction(PVModdedIcon("magnifying-glass"), QLineEdit::LeadingPosition);
	connect(_search, &QLineEdit::textChanged, this, &PVSettingsDialog::set_filter);
	auto* find = new QShortcut(QKeySequence::Find, this);
	connect(find, &QShortcut::activated, this, [this]() {
		_search->setFocus();
		_search->selectAll();
	});

	_count = new QLabel;
	_count->setObjectName("count");

	_json_toggle = new QToolButton;
	_json_toggle->setObjectName("json_toggle");
	_json_toggle->setCheckable(true);
	_json_toggle->setIcon(PVModdedIcon("brackets-curly"));
	_json_toggle->setText(tr("Edit as JSON"));
	_json_toggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
	_json_toggle->setToolTip(tr("Edit settings.json beside the default settings"));
	connect(_json_toggle, &QToolButton::toggled, this, &PVSettingsDialog::set_json_mode);

	auto* top = new QHBoxLayout;
	top->addWidget(_search, 1);
	top->addWidget(_count);
	top->addWidget(_json_toggle);

	_pages = new QStackedWidget;
	build_form();
	build_json_page();

	_notice = new QLabel;
	_notice->setObjectName("notice");
	_notice->setWordWrap(true);
	_notice->hide();

	_buttons = new QDialogButtonBox(QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok |
	                                QDialogButtonBox::Cancel | QDialogButtonBox::Apply);
	QPushButton* restore = _buttons->button(QDialogButtonBox::RestoreDefaults);
	restore->setText(tr("Reset the listed settings"));
	restore->setToolTip(tr("Give the settings listed their default values"));
	connect(restore, &QPushButton::clicked, this, &PVSettingsDialog::restore_listed_defaults);
	connect(_buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked, this,
	        [this]() { apply(); });
	connect(_buttons, &QDialogButtonBox::accepted, this, &PVSettingsDialog::accept);
	connect(_buttons, &QDialogButtonBox::rejected, this, &PVSettingsDialog::reject);

	auto* layout = new QVBoxLayout(this);
	layout->addLayout(top);
	layout->addWidget(_pages, 1);
	layout->addWidget(_notice);
	layout->addWidget(_buttons);

	refresh_rows();
	update_visibility();

	if (const QString error = settings.user_file_error(); not error.isEmpty()) {
		// Show the file as it is, and what is wrong with it, rather than settings
		// it does not hold.
		set_json_mode(true);
		QFile file(settings.user_file());
		if (file.open(QIODevice::ReadOnly)) {
			_user_json->setPlainText(QString::fromUtf8(file.readAll()));
		}
		check_json();
		show_notice(
		    tr("%1 cannot be read (%2): fix it here, or Squey goes on ignoring every setting "
		       "it holds.")
		        .arg(settings.user_file(), error));
	}
	_search->setFocus();
}

PVGuiQt::PVSettingsDialog::~PVSettingsDialog() = default;

void PVGuiQt::PVSettingsDialog::build_form()
{
	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	const std::vector<PVCore::PVSettingDescriptor> descriptors = settings.descriptors();

	// The declared sections in their order, then those only the keys name.
	std::vector<QString> order;
	QHash<QString, QString> titles;
	for (PVCore::PVSettingsSection const& section : settings.sections()) {
		order.push_back(section.key);
		titles[section.key] = section.title;
		_descriptions[section.key] = section.description;
	}
	for (PVCore::PVSettingDescriptor const& descriptor : descriptors) {
		const QStringList components = section_of(descriptor.key).split('.');
		for (qsizetype count = 1; count <= components.size(); ++count) {
			const QString section = components.first(count).join('.');
			if (not titles.contains(section)) {
				order.push_back(section);
				titles[section] = settings.section_title(section);
			}
		}
	}

	QHash<QString, std::vector<PVCore::PVSettingDescriptor>> by_section;
	for (PVCore::PVSettingDescriptor const& descriptor : descriptors) {
		by_section[section_of(descriptor.key)].push_back(descriptor);
	}

	// A section is under the longest section that starts its key.
	QHash<QString, std::vector<QString>> children;
	for (QString const& section : order) {
		QString parent = section;
		do {
			parent = parent.contains('.') ? parent.section('.', 0, -2) : QString();
		} while (not parent.isEmpty() and not titles.contains(parent));
		children[parent].push_back(section);
	}

	_sections = new QTreeWidget;
	_sections->setObjectName("sections");
	_sections->setHeaderHidden(true);
	_sections->setMinimumWidth(200);
	std::function<bool(QString const&, QTreeWidgetItem*, QString const&)> add =
	    [&](QString const& section, QTreeWidgetItem* parent_item, QString const& parent_title) {
		    const QString full_title =
		        parent_title.isEmpty() ? titles[section] : parent_title + " › " + titles[section];
		    auto* item = parent_item ? new QTreeWidgetItem(parent_item) : new QTreeWidgetItem;
		    item->setText(0, titles[section]);
		    item->setToolTip(0, full_title);
		    item->setData(0, SECTION_KEY_ROLE, section);
		    _full_titles[section] = full_title;
		    _display_sections.push_back(section);
		    bool has_settings = by_section.contains(section);
		    for (QString const& child : children.value(section)) {
			    has_settings |= add(child, item, full_title);
		    }
		    if (not has_settings) {
			    // A section of the table of contents that no library filled.
			    delete item;
			    _display_sections.erase(
			        std::find(_display_sections.begin(), _display_sections.end(), section));
		    } else if (not parent_item) {
			    _sections->addTopLevelItem(item);
		    }
		    return has_settings;
	    };
	for (QString const& section : children.value(QString())) {
		add(section, nullptr, QString());
	}
	_sections->expandAll();

	_form = new QScrollArea;
	_form->setObjectName("form");
	_form->setWidgetResizable(true);
	auto* guard = new wheel_guard(_form);
	auto* rows = new QWidget;
	auto* layout = new QVBoxLayout(rows);
	for (QString const& section : _display_sections) {
		auto it = by_section.find(section);
		const QString description = _descriptions.value(section);
		if (it == by_section.end() and description.isEmpty()) {
			continue;
		}
		auto* header = new QWidget;
		header->setObjectName("section:" + section);
		auto* header_layout = new QVBoxLayout(header);
		header_layout->setContentsMargins(0, 12, 0, 0);
		auto* title = new QLabel(_full_titles[section]);
		QFont font = title->font();
		font.setPointSizeF(font.pointSizeF() * 1.3);
		font.setBold(true);
		title->setFont(font);
		header_layout->addWidget(title);
		if (not description.isEmpty()) {
			auto* text = new QLabel(description);
			text->setWordWrap(true);
			header_layout->addWidget(text);
		}
		layout->addWidget(header);
		_headers[section] = header;
		if (it == by_section.end()) {
			continue;
		}
		for (PVCore::PVSettingDescriptor const& descriptor : *it) {
			auto* row = new __impl::setting_row(descriptor, guard, rows);
			const QString key = descriptor.key;
			row->edited = [this, key](QVariant const& value) { set_draft_value(key, value); };
			row->reset = [this, key]() { reset_draft_value(key); };
			layout->addWidget(row);
			_rows.push_back(row);
		}
	}
	_no_match = new QLabel(tr("No setting matches the search."));
	_no_match->setObjectName("no_match");
	_no_match->setAlignment(Qt::AlignCenter);
	layout->addWidget(_no_match);
	layout->addStretch();
	_form->setWidget(rows);

	connect(_sections, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem* item) {
		const QString section = item->data(0, SECTION_KEY_ROLE).toString();
		// The section itself, or the first of those under it.
		for (QString const& shown : _display_sections) {
			QWidget* header = _headers.value(shown);
			if (header and header->isVisible() and
			    (shown == section or shown.startsWith(section + '.'))) {
				_form->verticalScrollBar()->setValue(header->y());
				return;
			}
		}
	});

	auto* splitter = new QSplitter;
	splitter->addWidget(_sections);
	splitter->addWidget(_form);
	splitter->setStretchFactor(1, 1);
	splitter->setSizes({260, 840});
	_pages->insertWidget(FORM_PAGE, splitter);
}

void PVGuiQt::PVSettingsDialog::build_json_page()
{
	auto json_editor = [](QString const& name, QString const& syntax) {
		auto* editor = new QTextEdit;
		editor->setObjectName(name);
		editor->setAcceptRichText(false);
		editor->setLineWrapMode(QTextEdit::NoWrap);
		style_code_editor(editor, syntax);
		return editor;
	};

	// JSONC: the defaults carry their descriptions as comments.
	_defaults_json = json_editor("defaults_json", "JSONC");
	_defaults_json->setReadOnly(true);
	_defaults_json->setPlainText(defaults_document());
	_user_json = json_editor("user_json", "JSON");

	auto pane = [](QString const& title, QWidget* editor) {
		auto* widget = new QWidget;
		auto* layout = new QVBoxLayout(widget);
		layout->setContentsMargins(0, 0, 0, 0);
		auto* label = new QLabel(title);
		label->setTextInteractionFlags(Qt::TextSelectableByMouse);
		layout->addWidget(label);
		layout->addWidget(editor, 1);
		return widget;
	};
	auto* editors = new QSplitter;
	editors->addWidget(pane(tr("Default settings"), _defaults_json));
	editors->addWidget(pane(PVCore::PVSettings::get().user_file(), _user_json));
	editors->setStretchFactor(0, 3);
	editors->setStretchFactor(1, 2);

	_problems = new QListWidget;
	_problems->setObjectName("problems");
	_problems->setMaximumHeight(110);
	connect(_problems, &QListWidget::itemClicked, this, [this](QListWidgetItem* item) {
		const int line = item->data(LINE_ROLE).toInt();
		if (line > 0) {
			_user_json->setTextCursor(
			    QTextCursor(_user_json->document()->findBlockByNumber(line - 1)));
			_user_json->setFocus();
		}
	});

	_json_help = new QLabel;
	_json_help->setObjectName("json_help");
	_json_help->setWordWrap(true);
	_json_help->setTextFormat(Qt::RichText);

	_check_timer = new QTimer(this);
	_check_timer->setSingleShot(true);
	_check_timer->setInterval(300);
	connect(_check_timer, &QTimer::timeout, this, &PVSettingsDialog::check_json);
	connect(_user_json, &QTextEdit::textChanged, _check_timer, qOverload<>(&QTimer::start));
	connect(_user_json, &QTextEdit::cursorPositionChanged, this,
	        [this]() { show_help_at_cursor(_user_json); });
	connect(_defaults_json, &QTextEdit::cursorPositionChanged, this,
	        [this]() { show_help_at_cursor(_defaults_json); });

	auto* page = new QWidget;
	auto* layout = new QVBoxLayout(page);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(editors, 1);
	layout->addWidget(_problems);
	layout->addWidget(_json_help);
	_pages->insertWidget(JSON_PAGE, page);
}

QString PVGuiQt::PVSettingsDialog::defaults_document() const
{
	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	// For the pane to show the comments whole.
	constexpr int WIDTH = 60;
	const QString INDENT = "    ";
	auto comment = [&](QStringList& lines, QString const& text) {
		for (QString const& line : wrapped(text, WIDTH)) {
			lines << INDENT + "// " + line;
		}
	};

	QHash<QString, std::vector<PVCore::PVSettingDescriptor>> by_section;
	for (PVCore::PVSettingDescriptor const& descriptor : settings.descriptors()) {
		by_section[section_of(descriptor.key)].push_back(descriptor);
	}
	size_t remaining = 0;
	for (QString const& section : _display_sections) {
		remaining += by_section.value(section).size();
	}

	QStringList lines{"// The default settings. Copy those to change into",
	                  "// settings.json, on the right.", "{"};
	for (QString const& section : _display_sections) {
		const QString description = _descriptions.value(section);
		if (not by_section.contains(section) and description.isEmpty()) {
			continue;
		}
		if (lines.size() > 3) {
			lines << "";
		}
		lines << INDENT + "// --- " + _full_titles[section] + " ---";
		if (not description.isEmpty()) {
			comment(lines, description);
		}
		for (PVCore::PVSettingDescriptor const& descriptor : by_section.value(section)) {
			lines << "" << INDENT + "// " + descriptor.title;
			QString explanation = descriptor.description;
			if (const QString accepted = accepted_values(descriptor); not accepted.isEmpty()) {
				explanation += " " + accepted;
			}
			if (descriptor.requires_restart) {
				explanation += " Applies after a restart.";
			}
			comment(lines, explanation);
			--remaining;
			lines << INDENT + QString("\"%1\": %2%3")
			                      .arg(descriptor.key,
			                           json_text(PVCore::PVSettings::to_json(
			                               descriptor, settings.default_value(descriptor.key))),
			                           remaining > 0 ? "," : "");
		}
	}
	lines << "}";
	return lines.join('\n') + '\n';
}

void PVGuiQt::PVSettingsDialog::set_filter(QString const& filter)
{
	if (_search->text() != filter) {
		_search->setText(filter);
		return;
	}
	_filter_words.clear();
	_modified_only = false;
	for (QString const& word : filter.split(' ', Qt::SkipEmptyParts)) {
		if (word.compare("@modified", Qt::CaseInsensitive) == 0) {
			_modified_only = true;
		} else {
			_filter_words << word;
		}
	}
	update_visibility();
}

void PVGuiQt::PVSettingsDialog::set_json_mode(bool json)
{
	if (json != json_mode()) {
		if (json) {
			_user_json->setPlainText(document_text(_draft));
			check_json();
			_pages->setCurrentIndex(JSON_PAGE);
		} else if (take_json()) {
			refresh_rows();
			update_visibility();
			_pages->setCurrentIndex(FORM_PAGE);
			_buttons->button(QDialogButtonBox::Ok)->setEnabled(true);
			_buttons->button(QDialogButtonBox::Apply)->setEnabled(true);
		}
	}
	// The search filters the list, which the JSON hides.
	_search->setEnabled(not json_mode());
	_buttons->button(QDialogButtonBox::RestoreDefaults)->setEnabled(not json_mode());
	const QSignalBlocker blocker(_json_toggle);
	_json_toggle->setChecked(json_mode());
}

bool PVGuiQt::PVSettingsDialog::json_mode() const
{
	return _pages->currentIndex() == JSON_PAGE;
}

QJsonObject PVGuiQt::PVSettingsDialog::draft() const
{
	return _draft;
}

bool PVGuiQt::PVSettingsDialog::apply()
{
	if (json_mode() and not take_json()) {
		return false;
	}

	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	QStringList changed;
	const QMetaObject::Connection connection =
	    connect(&settings, &PVCore::PVSettings::changed, this,
	            [&changed](QString const& key) { changed << key; });
	QString error;
	const bool saved = settings.set_user_settings(_draft, &error);
	disconnect(connection);
	if (not saved) {
		show_notice(error);
		return false;
	}

	_restart_needed = false;
	for (QString const& key : changed) {
		const auto descriptor = settings.descriptor(key);
		_restart_needed |= descriptor and descriptor->requires_restart;
	}
	show_notice(_restart_needed ? tr("Saved. Some of the changes apply after a restart of Squey.")
	                            : QString());
	refresh_rows();
	return true;
}

void PVGuiQt::PVSettingsDialog::accept()
{
	if (not apply()) {
		return;
	}
	if (_restart_needed) {
		// Not modal, as the dialog that said it is gone.
		auto* box = new QMessageBox(QMessageBox::Information, windowTitle(),
		                            tr("Some of the changes apply after a restart of Squey."),
		                            QMessageBox::Ok, parentWidget());
		box->setAttribute(Qt::WA_DeleteOnClose);
		box->open();
	}
	QDialog::accept();
}

void PVGuiQt::PVSettingsDialog::refresh_rows()
{
	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	for (__impl::setting_row* row : _rows) {
		PVCore::PVSettingDescriptor const& descriptor = row->descriptor();
		const bool locked = settings.is_locked(descriptor.key);
		QVariant value =
		    locked ? settings.value(descriptor.key) : settings.default_value(descriptor.key);
		QString problem;
		if (not locked and _draft.contains(descriptor.key)) {
			QString error;
			if (auto draft_value =
			        PVCore::PVSettings::from_json(descriptor, _draft[descriptor.key], &error)) {
				value = *draft_value;
			} else {
				problem = tr("The value %1 of settings.json is ignored: %2.")
				              .arg(json_text(_draft[descriptor.key]), error);
			}
		}
		row->show_value(value, _draft.contains(descriptor.key), locked, problem);
	}
}

void PVGuiQt::PVSettingsDialog::update_visibility()
{
	QSet<QString> shown_sections;
	int shown = 0;
	for (__impl::setting_row* row : _rows) {
		const bool matches = (not _modified_only or row->is_modified()) and
		                     row->matches(_filter_words, _full_titles.value(row->section()));
		row->setVisible(matches);
		if (matches) {
			++shown;
			shown_sections.insert(row->section());
		}
	}
	for (auto it = _headers.begin(); it != _headers.end(); ++it) {
		const QString& section = it.key();
		it.value()->setVisible(
		    std::any_of(shown_sections.begin(), shown_sections.end(), [&](QString const& shown) {
			    return shown == section or shown.startsWith(section + '.');
		    }));
	}
	_no_match->setVisible(shown == 0);
	_count->setText(shown == 1 ? tr("1 setting") : tr("%1 settings").arg(shown));

	std::function<bool(QTreeWidgetItem*)> show_item = [&](QTreeWidgetItem* item) {
		bool any = shown_sections.contains(item->data(0, SECTION_KEY_ROLE).toString());
		for (int child = 0; child < item->childCount(); ++child) {
			any |= show_item(item->child(child));
		}
		item->setHidden(not any);
		return any;
	};
	for (int item = 0; item < _sections->topLevelItemCount(); ++item) {
		show_item(_sections->topLevelItem(item));
	}
}

void PVGuiQt::PVSettingsDialog::set_draft_value(QString const& key, QVariant const& value)
{
	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	const auto descriptor = settings.descriptor(key);
	if (not descriptor) {
		return;
	}
	if (value == settings.default_value(key)) {
		_draft.remove(key);
	} else {
		_draft[key] = PVCore::PVSettings::to_json(*descriptor, value);
	}
	refresh_rows();
}

void PVGuiQt::PVSettingsDialog::reset_draft_value(QString const& key)
{
	_draft.remove(key);
	refresh_rows();
}

void PVGuiQt::PVSettingsDialog::restore_listed_defaults()
{
	for (__impl::setting_row* row : _rows) {
		if (not row->isHidden()) {
			_draft.remove(row->descriptor().key);
		}
	}
	refresh_rows();
}

void PVGuiQt::PVSettingsDialog::check_json()
{
	_problems->clear();
	const QString text = _user_json->toPlainText();
	const QByteArray utf8 = text.toUtf8();
	QJsonParseError parse_error;
	const QJsonDocument document = QJsonDocument::fromJson(utf8, &parse_error);
	const bool parses = parse_error.error == QJsonParseError::NoError and document.isObject();
	_buttons->button(QDialogButtonBox::Ok)->setEnabled(parses);
	_buttons->button(QDialogButtonBox::Apply)->setEnabled(parses);

	auto add_problem = [this](QStyle::StandardPixmap icon, int line, QString const& text) {
		auto* item = new QListWidgetItem(style()->standardIcon(icon),
		                                 line > 0 ? tr("Line %1: %2").arg(line).arg(text) : text);
		item->setData(LINE_ROLE, line);
		_problems->addItem(item);
	};
	if (parse_error.error != QJsonParseError::NoError) {
		const int line = QString::fromUtf8(utf8.left(parse_error.offset)).count('\n') + 1;
		add_problem(QStyle::SP_MessageBoxCritical, line, parse_error.errorString());
		return;
	}
	if (not document.isObject()) {
		add_problem(QStyle::SP_MessageBoxCritical, 1,
		            tr("The settings are a JSON object, between { and }"));
		return;
	}
	std::vector<std::pair<int, PVCore::PVSettingsProblem>> problems;
	for (PVCore::PVSettingsProblem const& problem :
	     PVCore::PVSettings::get().check(document.object())) {
		problems.emplace_back(line_of_key(text, problem.key), problem);
	}
	std::stable_sort(problems.begin(), problems.end(),
	                 [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
	for (auto const& [line, problem] : problems) {
		add_problem(problem.severity == PVCore::PVSettingsProblem::Severity::Error
		                ? QStyle::SP_MessageBoxCritical
		                : QStyle::SP_MessageBoxWarning,
		            line, QString("%1: %2").arg(problem.key, problem.message));
	}
}

void PVGuiQt::PVSettingsDialog::show_help_at_cursor(QTextEdit* editor)
{
	static const QRegularExpression KEY(R"re("([^"]+)"\s*:)re");
	const QRegularExpressionMatch match = KEY.match(editor->textCursor().block().text());
	if (not match.hasMatch()) {
		_json_help->clear();
		return;
	}
	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	const QString key = match.captured(1);
	const auto descriptor = settings.descriptor(key);
	if (not descriptor) {
		_json_help->setText(
		    tr("<code>%1</code> is not a setting of Squey.").arg(key.toHtmlEscaped()));
		return;
	}
	QString help =
	    QString("<b>%1</b>: %2")
	        .arg(descriptor->title.toHtmlEscaped(), descriptor->description.toHtmlEscaped());
	if (const QString accepted = accepted_values(*descriptor); not accepted.isEmpty()) {
		help += " " + accepted.toHtmlEscaped();
	}
	help +=
	    " " +
	    tr("Default: <code>%1</code>.")
	        .arg(json_text(PVCore::PVSettings::to_json(*descriptor, settings.default_value(key)))
	                 .toHtmlEscaped());
	if (descriptor->requires_restart) {
		help += " " + tr("Applies after a restart.");
	}
	_json_help->setText(help);
}

bool PVGuiQt::PVSettingsDialog::take_json()
{
	const QByteArray utf8 = _user_json->toPlainText().toUtf8();
	QJsonParseError parse_error;
	const QJsonDocument document = QJsonDocument::fromJson(utf8, &parse_error);
	if (parse_error.error != QJsonParseError::NoError or not document.isObject()) {
		check_json();
		show_notice(tr("The JSON does not parse: see the problem under it."));
		return false;
	}
	_draft = document.object();
	show_notice(QString());
	return true;
}

void PVGuiQt::PVSettingsDialog::show_notice(QString const& text)
{
	_notice->setText(text);
	_notice->setVisible(not text.isEmpty());
}
