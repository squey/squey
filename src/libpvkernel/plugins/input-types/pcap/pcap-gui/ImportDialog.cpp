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

#include "ImportDialog.h"

#include "Background.h"
#include "CapturesPanel.h"
#include "ColumnsPanel.h"
#include "FieldsPanel.h"
#include "FilterPanel.h"
#include "OptionsPanel.h"
#include "PreviewPanel.h"
#include "ProfileEditor.h"
#include "ProfilesPanel.h"
#include "Style.h"

#include <libpvpcap.h>
#include <libpvpcap/profileformat.h>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVLogger.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMessageBox>
#include <QMimeData>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QShortcut>
#include <QSplitter>
#include <QStyle>
#include <QTabWidget>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>

namespace PVPcapsicum
{

namespace
{

const QString LAST_PROFILE_KEY = "pcap/last_profile";
const QString LAST_PROFILE_BUILTIN_KEY = "pcap/last_profile_builtin";
const QString GEOMETRY_KEY = "pcap/import_dialog_geometry";
const QString SPLITTERS_KEY = "pcap/import_dialog_splitters";

//! Of the tab of the options of the profile, after that of its columns.
constexpr int OPTIONS_TAB = 1;

//! What tshark lists of its fields, and its loading told.
struct loaded_catalog {
	std::shared_ptr<const pvpcap::field_catalog> catalog;
	QStringList names; //!< of the fields and the protocols, sorted
	QString problem;   //!< why there is no catalog
};

//! Shared by the dialogs open at once, and read again once none is.
std::weak_ptr<const loaded_catalog>& cached_catalog()
{
	static std::weak_ptr<const loaded_catalog> catalog;
	return catalog;
}

std::shared_ptr<const loaded_catalog> load()
{
	auto loaded = std::make_shared<loaded_catalog>();
	try {
		loaded->catalog = pvpcap::field_catalog::load();
	} catch (const std::exception& e) {
		loaded->problem = QString::fromStdString(e.what());
		return loaded;
	}
	QStringList& names = loaded->names;
	names.reserve(
	    qsizetype(loaded->catalog->fields().size() + loaded->catalog->protocols().size()));
	for (const auto& protocol : loaded->catalog->protocols()) {
		names << style::to_qstring(protocol.filter_name);
	}
	for (const auto& field : loaded->catalog->fields()) {
		names << style::to_qstring(field.filter_name);
	}
	names.sort(Qt::CaseInsensitive);
	names.removeDuplicates();
	return loaded;
}

QWidget* in_widget(QLayout* layout)
{
	auto* widget = new QWidget;
	layout->setContentsMargins(0, 0, 0, 0);
	widget->setLayout(layout);
	return widget;
}

} // namespace

ImportDialog::ImportDialog(const QStringList& captures, QWidget* parent)
    : QDialog(parent)
    , _editor(new ProfileEditor(this))
    , _catalog_job(new BackgroundJob(this))
    , _filter_job(new BackgroundJob(this))
    , _filter_waiting(new QTimer(this))
{
	setWindowTitle(tr("Import captures"));
	setAcceptDrops(true);
	build();
	connect_parts();

	const QString name = PVCore::PVConfig::value(LAST_PROFILE_KEY).toString();
	const QVariant builtin = PVCore::PVConfig::value(LAST_PROFILE_BUILTIN_KEY);
	// Earlier versions only kept the name: a profile of the user is more likely.
	_editor->load({builtin.isValid() and builtin.toBool(), name.toStdString()});
	if (not builtin.isValid() and _editor->current().name != name.toStdString()) {
		const pvpcap::profile_location shipped{true, name.toStdString()};
		const auto& profiles = _editor->profiles();
		if (std::find(profiles.begin(), profiles.end(), shipped) != profiles.end()) {
			_editor->choose(shipped);
		}
	}

	load_catalog();
	_captures->add(captures);
	show_profile();
}

ImportDialog::~ImportDialog() = default;

void ImportDialog::build()
{
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(12, 12, 12, 10);
	layout->setSpacing(10);

	_captures = new CapturesPanel;
	_profiles = new ProfilesPanel(_editor);
	auto* sidebar = new QVBoxLayout;
	sidebar->setSpacing(14);
	sidebar->addWidget(_captures, 3);
	sidebar->addWidget(_profiles, 2);

	_profile_name = new QLabel;
	_profile_name->setObjectName("profile_name");
	QFont title = _profile_name->font();
	title.setPointSizeF(title.pointSizeF() * 1.35);
	title.setBold(true);
	_profile_name->setFont(title);
	_profile_state = new QLabel;
	_profile_state->setObjectName("profile_state");
	_revert = new QPushButton(PVModdedIcon("arrow-rotate-left"), tr("Revert"));
	_revert->setObjectName("revert_profile_button");
	_revert->setToolTip(tr("Drop the changes of the profile"));
	_save = new QPushButton(PVModdedIcon("floppy-disk"), tr("Save"));
	_save->setObjectName("save_profile_button");
	_save->setToolTip(tr("Save the changes of the profile (Ctrl+S)"));
	_save_as = new QPushButton(PVModdedIcon("floppy-disk-circle-arrow-right"), tr("Save as…"));
	_save_as->setObjectName("save_profile_as_button");
	_save_as->setToolTip(tr("Save the profile, as changed, as a new profile of yours "
	                        "(Ctrl+Shift+S)"));
	for (QPushButton* button : {_revert, _save, _save_as}) {
		button->setAutoDefault(false);
	}
	auto* header = new QHBoxLayout;
	header->setSpacing(10);
	header->addWidget(_profile_name);
	header->addWidget(_profile_state, 1);
	header->addWidget(_revert);
	header->addWidget(_save);
	header->addWidget(_save_as);

	_fields = new FieldsPanel;
	_columns = new ColumnsPanel;
	_choice = new QSplitter(Qt::Horizontal);
	_choice->setObjectName("choice_splitter");
	_choice->addWidget(_fields);
	_choice->addWidget(_columns);
	_choice->setStretchFactor(0, 3);
	_choice->setStretchFactor(1, 2);
	_choice->setChildrenCollapsible(false);

	auto* choosing = new QWidget;
	auto* choosing_layout = new QVBoxLayout(choosing);
	choosing_layout->setContentsMargins(8, 8, 8, 8);
	choosing_layout->addWidget(_choice);
	_options = new OptionsPanel;
	// The tab may be lower than the options.
	auto* options = new QScrollArea;
	options->setWidget(_options);
	options->setWidgetResizable(true);
	options->setFrameShape(QFrame::NoFrame);
	_tabs = new QTabWidget;
	_tabs->setObjectName("profile_tabs");
	_tabs->addTab(choosing, tr("&Columns"));
	_tabs->addTab(options, tr("&Options"));
	_filter = new FilterPanel;
	auto* editing = new QVBoxLayout;
	editing->setSpacing(8);
	editing->addWidget(_tabs, 1);
	editing->addWidget(_filter);

	_preview = new PreviewPanel;
	_vertical = new QSplitter(Qt::Vertical);
	_vertical->setObjectName("preview_splitter");
	_vertical->addWidget(in_widget(editing));
	_vertical->addWidget(_preview);
	_vertical->setStretchFactor(0, 3);
	_vertical->setStretchFactor(1, 2);
	_vertical->setChildrenCollapsible(false);

	auto* right = new QVBoxLayout;
	right->setSpacing(8);
	right->addLayout(header);
	right->addWidget(_vertical, 1);

	_main = new QSplitter(Qt::Horizontal);
	_main->setObjectName("main_splitter");
	_main->addWidget(in_widget(sidebar));
	_main->addWidget(in_widget(right));
	_main->setStretchFactor(0, 1);
	_main->setStretchFactor(1, 4);
	_main->setChildrenCollapsible(false);
	_main->setHandleWidth(14);
	layout->addWidget(_main, 1);

	_summary = new QLabel;
	_summary->setObjectName("import_summary");
	_problem = new QLabel;
	_problem->setObjectName("import_problem");
	style::color_label(_problem, style::warning());
	auto* cancel = new QPushButton(tr("Cancel"));
	cancel->setObjectName("cancel_button");
	cancel->setAutoDefault(false);
	_import = new QPushButton(tr("&Import"));
	_import->setObjectName("process_import_button");
	_import->setDefault(true);
	style::make_primary(_import);
	auto* footer = new QHBoxLayout;
	footer->setSpacing(12);
	footer->addWidget(_summary);
	footer->addWidget(_problem, 1);
	footer->addWidget(cancel);
	footer->addWidget(_import);
	layout->addLayout(footer);
	connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
	connect(_import, &QPushButton::clicked, this, &QDialog::accept);

	const QRect screen =
	    (parentWidget() ? parentWidget()->screen() : QGuiApplication::primaryScreen())
	        ->availableGeometry();
	const QByteArray geometry = PVCore::PVConfig::value(GEOMETRY_KEY).toByteArray();
	if (geometry.isEmpty() or not restoreGeometry(geometry)) {
		resize(std::min(1360, screen.width() * 9 / 10), std::min(880, screen.height() * 9 / 10));
		_main->setSizes({290, 1000});
		_vertical->setSizes({540, 300});
	}
	const QList<QVariant> splitters = PVCore::PVConfig::value(SPLITTERS_KEY).toList();
	if (splitters.size() == 3) {
		_main->restoreState(splitters[0].toByteArray());
		_choice->restoreState(splitters[1].toByteArray());
		_vertical->restoreState(splitters[2].toByteArray());
	}

	new QShortcut(QKeySequence::Find, this, [this]() { _fields->focus_search(); });
	new QShortcut(QKeySequence::Open, this, [this]() { _captures->ask(); });
	new QShortcut(QKeySequence::Save, this, [this]() {
		if (_editor->current().builtin) {
			save_as();
		} else if (_editor->modified()) {
			save();
		}
	});
	new QShortcut(QKeySequence::SaveAs, this, [this]() { save_as(); });
}

void ImportDialog::connect_parts()
{
	connect(_editor, &ProfileEditor::profile_changed, this, &ImportDialog::show_profile);
	connect(_editor, &ProfileEditor::current_changed, this, &ImportDialog::update_header);
	connect(_editor, &ProfileEditor::profiles_changed, this, &ImportDialog::update_header);

	connect(_captures, &CapturesPanel::captures_changed, this, [this]() {
		_preview->preview(_captures->captures(), _editor->profile());
		update_summary();
	});
	connect(_captures, &CapturesPanel::infos_changed, this, &ImportDialog::update_summary);
	connect(_captures, &CapturesPanel::samples_changed, this, &ImportDialog::show_samples);

	connect(_fields, &FieldsPanel::toggled, this, &ImportDialog::toggle_field);
	connect(_columns, &ColumnsPanel::edited, this,
	        [this](const std::vector<pvpcap::profile_field>& columns) { set_columns(columns); });
	connect(_columns, &ColumnsPanel::dropped, this, &ImportDialog::insert_fields);
	connect(_columns, &ColumnsPanel::reveal, _fields, &FieldsPanel::reveal);
	const auto change = [this](const pvpcap::profile& edited) {
		pvpcap::profile p = edited;
		p.columns = _editor->profile().columns;
		_editor->change(p);
	};
	connect(_filter, &FilterPanel::edited, this, change);
	connect(_options, &OptionsPanel::edited, this, change);
	connect(_preview, &PreviewPanel::previewed, this,
	        [this](const std::map<std::string, double>& filled, size_t rows, uint64_t frames,
	               const QString& capture) {
		        _columns->set_filled(filled);
		        _filter->set_note(frames == 0
		                              ? QString()
		                              : tr("It keeps %1 of the first %2 frames of %3.")
		                                    .arg(style::count_text(rows), style::count_text(frames),
		                                         QFileInfo(capture).fileName()));
	        });

	_filter_waiting->setSingleShot(true);
	_filter_waiting->setInterval(250);
	connect(_filter_waiting, &QTimer::timeout, this, &ImportDialog::check_filter);
	connect(_revert, &QPushButton::clicked, _editor, &ProfileEditor::revert);
	connect(_save, &QPushButton::clicked, this, &ImportDialog::save);
	connect(_save_as, &QPushButton::clicked, this, &ImportDialog::save_as);
}

void ImportDialog::load_catalog()
{
	const auto show = [this](const std::shared_ptr<const loaded_catalog>& loaded) {
		_catalog = loaded->catalog;
		_fields->set_catalog(_catalog, loaded->problem);
		_columns->set_catalog(_catalog);
		_filter->set_names(loaded->names);
		update_summary();
	};
	if (const std::shared_ptr<const loaded_catalog> cached = cached_catalog().lock()) {
		_loaded = cached;
		show(cached);
		return;
	}
	_catalog_job->run([](const std::atomic<bool>&) { return load(); },
	                  [this, show](std::shared_ptr<const loaded_catalog> loaded) {
		                  cached_catalog() = loaded;
		                  _loaded = loaded;
		                  show(loaded);
	                  });
}

void ImportDialog::show_profile()
{
	const pvpcap::profile& p = _editor->profile();
	_columns->set_columns(p.columns);
	_fields->set_columns(p.columns);
	_filter->set_profile(p);
	_options->set_profile(p);
	const QString problem = _options->problem();
	_tabs->setTabIcon(OPTIONS_TAB, problem.isEmpty()
	                                   ? QIcon()
	                                   : style()->standardIcon(QStyle::SP_MessageBoxWarning));
	_tabs->setTabToolTip(OPTIONS_TAB, problem);
	_captures->set_reading_options(pvpcap::ws_reading_options(p));
	_preview->preview(_captures->captures(), p);
	if (_filter_check and p.filter == _checked_filter) {
		_filter->set_check(_filter_check);
	} else if (p.filter != _pending_filter) {
		_pending_filter = p.filter;
		_filter_check.reset();
		_filter_job->cancel();
		_filter->set_check(std::nullopt);
		_filter_waiting->start();
	}
	update_header();
	update_summary();
}

void ImportDialog::show_samples()
{
	std::vector<const pvpcap::capture_sample*> samples;
	bool reading = false;
	for (const capture_info* info : _captures->infos()) {
		if (info->sample) {
			samples.push_back(&*info->sample);
		}
		reading |= info->reading;
	}
	sampled_fields merged = merge_samples(samples);
	_columns->set_samples(merged);
	_fields->set_samples(std::move(merged), reading);
}

void ImportDialog::check_filter()
{
	const std::string filter = _pending_filter.value_or(_editor->profile().filter);
	const auto checked = [this, filter](const pvpcap::filter_check& check) {
		_pending_filter.reset();
		_filter_check = check;
		_checked_filter = filter;
		if (filter == _editor->profile().filter) {
			_filter->set_check(check);
		}
		update_summary();
	};
	if (filter.empty()) {
		checked(pvpcap::filter_check{true, {}, 0, 0});
		return;
	}
	_filter_job->run([filter](const std::atomic<bool>&) { return pvpcap::ws_check_filter(filter); },
	                 checked);
}

void ImportDialog::update_header()
{
	const pvpcap::profile_location& location = _editor->current();
	const bool modified = _editor->modified();
	_profile_name->setText(QString::fromStdString(location.name));
	const QString problem = _editor->problem(location);
	if (not problem.isEmpty()) {
		style::color_label(_profile_state, style::error(), true);
		_profile_state->setText(tr("can't be read: %1").arg(problem));
	} else if (location.builtin) {
		style::color_label(_profile_state, modified ? style::accent() : style::secondary(), true);
		_profile_state->setText(modified ? tr("built-in, changed: save it as a profile of yours "
		                                      "to keep the changes")
		                                 : tr("built-in"));
	} else {
		style::color_label(_profile_state, style::accent(), true);
		_profile_state->setText(modified ? tr("changed, not saved") : QString());
	}
	_revert->setVisible(modified);
	_save->setVisible(modified and not location.builtin);
}

QStringList ImportDialog::two_pass_fields() const
{
	const pvpcap::profile& p = _editor->profile();
	QStringList fields;
	for (const pvpcap::profile_field& column : p.columns) {
		if (pvpcap::ws_needs_two_passes(column.name, column.type)) {
			fields << QString::fromStdString(column.name);
		}
	}
	if (_catalog) {
		for (const std::string& name : pvpcap::ws_filter_fields(p.filter)) {
			const int64_t field = _catalog->find_field(name);
			const std::string type(field >= 0 ? _catalog->fields()[field].type : "");
			if (pvpcap::ws_needs_two_passes(name, type) and
			    not fields.contains(QString::fromStdString(name))) {
				fields << QString::fromStdString(name);
			}
		}
	}
	return fields;
}

QString ImportDialog::problem() const
{
	const QStringList captures = _captures->captures();
	if (captures.isEmpty()) {
		return tr("Add the captures to import.");
	}
	if (_editor->profile().columns.empty()) {
		return tr("Choose the columns to import.");
	}
	QStringList unreadable;
	for (const capture_info* info : _captures->infos()) {
		if (not info->error.isEmpty()) {
			unreadable << QFileInfo(info->path).fileName();
		}
	}
	if (not unreadable.isEmpty()) {
		return tr("tshark can't read %1: remove it to import the others.")
		    .arg(unreadable.join(", "));
	}
	if (_filter_check and _checked_filter == _editor->profile().filter and
	    not _filter_check->problem.empty()) {
		return tr("The filter is wrong.");
	}
	return {};
}

void ImportDialog::update_summary()
{
	const std::vector<const capture_info*> infos = _captures->infos();
	uint64_t size = 0;
	uint64_t frames = 0;
	bool counted = true;
	bool estimated = false;
	for (const capture_info* info : infos) {
		size += info->overview ? info->overview->size : uint64_t(QFileInfo(info->path).size());
		if (info->overview and info->overview->estimated_frames != 0) {
			frames += info->overview->estimated_frames;
			estimated |= not info->overview->whole;
		} else {
			counted = false;
		}
	}
	QStringList parts;
	if (not infos.empty()) {
		parts << (infos.size() == 1 ? tr("1 capture") : tr("%1 captures").arg(infos.size()))
		      << style::size_text(size);
		if (counted) {
			parts << (estimated ? tr("≈ %1 frames").arg(style::short_count_text(frames))
			                    : tr("%1 frames").arg(style::count_text(frames)));
		}
	}
	const size_t columns = _editor->profile().columns.size();
	parts << (columns == 1 ? tr("1 column") : tr("%1 columns").arg(columns));
	const QStringList twice = two_pass_fields();
	if (not twice.isEmpty()) {
		parts << tr("read twice");
	}
	_summary->setText(parts.join(" · "));
	_summary->setToolTip(
	    twice.isEmpty()
	        ? QString()
	        : tr("tshark reads the captures twice, which takes about a quarter longer, for %1.")
	              .arg(twice.join(", ")));

	const QString why = problem();
	_problem->setText(why);
	_import->setEnabled(why.isEmpty());
}

void ImportDialog::set_columns(std::vector<pvpcap::profile_field> columns)
{
	pvpcap::profile p = _editor->profile();
	p.columns = std::move(columns);
	_editor->change(p);
}

void ImportDialog::toggle_field(const pvpcap::profile_field& field, bool column)
{
	std::vector<pvpcap::profile_field> columns = _editor->profile().columns;
	const auto it = std::find_if(columns.begin(), columns.end(),
	                             [&](const auto& c) { return c.name == field.name; });
	if (column and it == columns.end()) {
		columns.push_back(field);
	} else if (not column and it != columns.end()) {
		columns.erase(it);
	} else {
		return;
	}
	set_columns(std::move(columns));
}

void ImportDialog::insert_fields(const QStringList& names, int row)
{
	std::vector<pvpcap::profile_field> columns = _editor->profile().columns;
	for (const QString& name : names) {
		const std::string field = name.toStdString();
		const auto it = std::find_if(columns.begin(), columns.end(),
		                             [&](const auto& c) { return c.name == field; });
		std::string type;
		if (it != columns.end()) {
			type = it->type;
			row -= it - columns.begin() < row;
			columns.erase(it);
		} else if (_catalog and _catalog->find_field(field) >= 0) {
			type = _catalog->fields()[_catalog->find_field(field)].type;
		} else {
			continue;
		}
		row = std::clamp(row, 0, int(columns.size()));
		columns.insert(columns.begin() + row++, {field, type});
	}
	set_columns(std::move(columns));
}

void ImportDialog::save()
{
	try {
		_editor->save();
	} catch (const pvpcap::profile_error& e) {
		QMessageBox::warning(this, tr("Can't save the profile"), QString::fromStdString(e.what()));
	}
}

void ImportDialog::save_as()
{
	const QString proposed =
	    QString::fromStdString(pvpcap::free_profile_name(_editor->current().name + " copy"));
	bool ok = false;
	const QString name =
	    QInputDialog::getText(this, tr("Save the profile as"), tr("Name of the new profile:"),
	                          QLineEdit::Normal, proposed, &ok);
	if (not ok) {
		return;
	}
	try {
		_editor->save_as(name.toStdString());
	} catch (const pvpcap::profile_error& e) {
		QMessageBox::warning(this, tr("Can't save the profile"), QString::fromStdString(e.what()));
	}
}

QStringList ImportDialog::captures() const
{
	return _captures->captures();
}

QDomDocument ImportDialog::format() const
{
	return pvpcap::get_format(_accepted, size_t(captures().size()));
}

void ImportDialog::accept()
{
	const pvpcap::profile p = _editor->profile();
	if (not p.filter.empty() and (not _filter_check or _checked_filter != p.filter)) {
		// Checked now rather than refused by tshark once the import started.
		_filter_job->cancel();
		_pending_filter.reset();
		_filter_check = pvpcap::ws_check_filter(p.filter);
		_checked_filter = p.filter;
		_filter->set_check(_filter_check);
	}
	update_summary();
	if (not problem().isEmpty()) {
		if (_filter_check and not _filter_check->problem.empty()) {
			_filter->setFocus();
		}
		return;
	}
	// Said now rather than once the import started, and with where tshark was looked for.
	if (pvpcap::ws_get_version().empty()) {
		QMessageBox::critical(
		    this, tr("tshark can't be run"),
		    tr("Squey reads captures with tshark, which could not be run from '%1'.\n\n"
		       "The environment variable SQUEY_TSHARK_PATH can give its location.")
		        .arg(QString::fromStdString(pvpcap::tshark_path())));
		return;
	}
	_accepted = p;
	remember();
	QDialog::accept();
}

void ImportDialog::reject()
{
	remember();
	QDialog::reject();
}

void ImportDialog::remember()
{
	PVCore::PVConfig::set_value(LAST_PROFILE_KEY, QString::fromStdString(_editor->current().name));
	PVCore::PVConfig::set_value(LAST_PROFILE_BUILTIN_KEY, _editor->current().builtin);
	PVCore::PVConfig::set_value(GEOMETRY_KEY, saveGeometry());
	PVCore::PVConfig::set_value(
	    SPLITTERS_KEY,
	    QList<QVariant>{_main->saveState(), _choice->saveState(), _vertical->saveState()});
	try {
		_editor->write_drafts();
	} catch (const pvpcap::profile_error& e) {
		PVLOG_WARN("The changes of the pcap profiles could not be kept: %s\n", e.what());
	}
}

void ImportDialog::dragEnterEvent(QDragEnterEvent* event)
{
	if (event->mimeData()->hasUrls()) {
		event->acceptProposedAction();
	}
}

void ImportDialog::dropEvent(QDropEvent* event)
{
	QStringList paths;
	for (const QUrl& url : event->mimeData()->urls()) {
		if (url.isLocalFile()) {
			paths << url.toLocalFile();
		}
	}
	_captures->add(paths);
	event->acceptProposedAction();
}

} // namespace PVPcapsicum
