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

#include <PVImportController.h>

#include <PVMainWindow.h>

#include <pvguiqt/PVGuiQtSettings.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvguiqt/PVImportPage.h>
#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/rush/PVConverter.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVUnicodeSourceError.h>
#include <pvkernel/widgets/PVImportWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/common.h>
#include <squey/PVAxis.h>
#include <squey/PVMappingProperties.h>
#include <squey/PVRoot.h>
#include <squey/PVScaled.h>
#include <squey/PVScalingFilter.h>
#include <squey/PVScene.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <QLocale>
#include <QMenuBar>
#include <QShortcut>
#include <QThread>

#include <algorithm>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace App
{

/**
 * Forecast of the memory taken by the structures built on top of an imported source.
 *
 * Only the row count changes while the import runs, so everything that depends on the format
 * alone is computed once and each forecast then boils down to a multiplication.
 */
class PVMemoryConsumptionForecast
{
  public:
	explicit PVMemoryConsumptionForecast(const Squey::PVSource& src)
	{
		const size_t column_count = src.get_nraw_column_count();
		const PVRush::PVFormat& format = src.get_format();

		size_t computed_column_count = 0;
		for (size_t i = 0; i < column_count; i++) {
			const Squey::PVMappingProperties mapping_properties(format, PVCol(i));
			computed_column_count += mapping_properties.get_mapping_filter()->is_computed();
		}

		// Each zone holds both a zone tree and a zoomed zone tree, hence the factor 2.
		_fixed_bytes = 2 * column_count * sizeof(PVParallelView::PVZoneTree::PVBranch) * NBUCKETS;

		_bytes_per_row = computed_column_count * sizeof(Squey::PVScalingFilter::value_type) +
		                 column_count * sizeof(Squey::PVScaled::value_type) +
		                 2 * column_count * sizeof(PVRow);
	}

	/** Memory consumption increase once @p row_count rows have been imported, in bytes. */
	size_t operator()(size_t row_count) const { return _fixed_bytes + _bytes_per_row * row_count; }

  private:
	size_t _fixed_bytes;   //!< Part that does not depend on the number of rows.
	size_t _bytes_per_row; //!< Part that each imported row adds.
};

} // namespace App

namespace
{

constexpr std::chrono::milliseconds POLL_PERIOD{50};
constexpr std::chrono::seconds MEMORY_CHECK_PERIOD{1};

/** @p text, of an exception, as the rich text of a message. */
QString plain(const char* text)
{
	return QString::fromStdString(text).toHtmlEscaped().replace('\n', "<br>");
}

} // namespace

App::PVImportController::PVImportController(PVMainWindow& window) : _window(window)
{
	_poll_timer.setInterval(POLL_PERIOD);
	connect(&_poll_timer, &QTimer::timeout, this, &PVImportController::poll);

	connect(&page(), &PVGuiQt::PVImportPage::close_requested, this,
	        &PVImportController::close_page);
	connect(&projects(), &PVGuiQt::PVProjectsTabWidget::import_page_close_requested, this,
	        &PVImportController::close_page);
	connect(&page(), &PVGuiQt::PVImportPage::import_requested, this,
	        &PVImportController::import_content);
	connect(&page(), &PVGuiQt::PVImportPage::task_finished, this,
	        &PVImportController::task_finished);
}

App::PVImportController::~PVImportController()
{
	// The window going away with an import running, as the application quits.
	if (_preparing) {
		_preparing->cancel();
	}
	if (_job) {
		_job->cancel();
	}
	if (_worker != nullptr) {
		_worker->wait();
		delete _worker;
	}
}

PVGuiQt::PVProjectsTabWidget& App::PVImportController::projects() const
{
	return *_window._projects_tab_widget;
}

PVGuiQt::PVImportPage& App::PVImportController::page() const
{
	return projects().import_page();
}

bool App::PVImportController::busy() const
{
	return _busy or page().task_running();
}

void App::PVImportController::import(std::vector<PVRush::PVSourceDescription> sources)
{
	for (PVRush::PVSourceDescription& source : sources) {
		PVRush::PVInputType_p type = source.get_source_creator()->supported_type_lib();
		Import import{type->tab_name_of_inputs(source.get_inputs()), type->icon(), {}, {}};
		// Such as that of a recent source of Parquet files, which is not saved.
		const PVRush::PVFormat& format = source.get_format();
		if (format.get_full_path().isEmpty() and format.get_axes().size() == 0) {
			import.prepare = [type, source](PVCore::PVTaskProgress& progress)
			    -> std::optional<PVRush::PVSourceDescription> {
				PVRush::PVFormat made =
				    type->get_format_from_inputs(source.get_inputs(), &progress);
				if (progress.canceled()) {
					return std::nullopt;
				}
				PVRush::PVSourceDescription prepared = source;
				prepared.set_format(made);
				return prepared;
			};
		} else {
			import.description = std::move(source);
		}
		enqueue(std::move(import));
	}
}

void App::PVImportController::import_params(PVRush::PVInputType_p type, QString params)
{
	Import import{type->menu_input_name(), type->icon(), {}, {}};
	import.prepare =
	    [type,
	     params](PVCore::PVTaskProgress& progress) -> std::optional<PVRush::PVSourceDescription> {
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVFormat format;
		if (not type->create_source_description_params(params, inputs, format, progress)) {
			if (progress.canceled()) {
				return std::nullopt;
			}
			throw std::runtime_error(
			    tr("Imports of %1 take no parameters.").arg(type->menu_input_name()).toStdString());
		}
		return PVRush::PVSourceDescription(
		    inputs, PVRush::PVSourceCreatorFactory::get_by_input_type(type), format);
	};
	enqueue(std::move(import));
}

void App::PVImportController::enqueue(Import import)
{
	_queue.push_back(std::move(import));
	if (not busy()) {
		begin();
		start_next();
	}
}

bool App::PVImportController::edit(PVRush::PVInputType_p type, PVWidgets::PVImportWidget* content)
{
	if (busy()) {
		delete content;
		page().add_message(PVGuiQt::PVImportPage::Severity::information,
		                   tr("Another import can start once this one is over."));
		projects().show_import_page();
		return false;
	}

	_content_type = std::move(type);
	Squey::PVRoot& root = _window.get_root();
	_collections.clear();
	QStringList names;
	int current = -1;
	for (Squey::PVScene* scene : root.get_children()) {
		if (scene == root.current_scene()) {
			current = int(names.size());
		}
		_collections.push_back(scene);
		names << QString::fromStdString(scene->get_name());
	}
	page().clear_messages();
	page().set_content(content, tr("Import: %1").arg(_content_type->menu_input_name()),
	                   _content_type->icon(), names, current);
	projects().show_import_page();
	return true;
}

void App::PVImportController::import_content()
{
	std::vector<PVWidgets::PVImportWidget::source_t> sources;
	try {
		sources = page().content()->sources();
	} catch (const std::exception& e) {
		page().add_message(PVGuiQt::PVImportPage::Severity::critical, plain(e.what()));
		return;
	}
	// As a dialog whose checks failed, which tells why itself.
	if (sources.empty()) {
		return;
	}
	PVRush::PVSourceCreator_p creator =
	    PVRush::PVSourceCreatorFactory::get_by_input_type(_content_type);
	std::vector<PVRush::PVSourceDescription> descriptions;
	for (auto& [inputs, format] : sources) {
		descriptions.emplace_back(inputs, creator, format);
	}

	// The collections may have changed since the page listed them.
	Squey::PVRoot& root = _window.get_root();
	const int chosen = page().chosen_collection();
	const auto scenes = root.get_children();
	if (chosen >= 0 and
	    std::find(scenes.begin(), scenes.end(), _collections[chosen]) != scenes.end()) {
		root.select_scene(*_collections[chosen]);
	} else {
		_new_scene = &projects().project_new();
	}

	_content_import = true;
	page().set_content_shown(false);
	import(std::move(descriptions));
}

void App::PVImportController::close_page()
{
	if (_busy) {
		return;
	}
	if (page().task_running()) {
		_closing = true;
		page().cancel_task();
		return;
	}
	page().clear_content();
	page().clear_messages();
	projects().hide_import_page();
}

void App::PVImportController::task_finished()
{
	if (busy()) {
		return;
	}
	if (std::exchange(_closing, false)) {
		close_page();
	}
	if (not _queue.empty()) {
		begin();
		start_next();
	} else {
		Q_EMIT idle();
	}
}

void App::PVImportController::begin()
{
	_busy = true;
	_imported = false;
	// Back once these imports are over, if they are not its own.
	page().set_content_shown(false);
	page().clear_messages();
	page().set_closable(false);
	set_window_enabled(false);
}

void App::PVImportController::next()
{
	QMetaObject::invokeMethod(this, &PVImportController::start_next, Qt::QueuedConnection);
}

void App::PVImportController::start_next()
{
	if (_queue.empty()) {
		end();
		return;
	}
	Import import = std::move(_queue.front());
	_queue.pop_front();

	_name = import.name;
	page().set_title(tr("Import of %1").arg(_name), import.icon);
	projects().show_import_page();
	if (import.prepare) {
		prepare(std::move(import.prepare));
	} else {
		create(std::move(*import.description));
	}
}

void App::PVImportController::prepare(
    std::function<std::optional<PVRush::PVSourceDescription>(PVCore::PVTaskProgress&)> prepare)
{
	_state = State::preparing;
	auto progress = std::make_shared<PVCore::PVTaskProgress>();
	auto prepared = std::make_shared<std::optional<PVRush::PVSourceDescription>>();
	auto error = std::make_shared<QString>();
	_preparing = progress;
	page().start_progress(
	    tr("Reading the inputs..."),
	    {{tr("Cancel"), [progress]() { progress->cancel(); }, PVModdedIcon("circle-x")}});

	_worker = QThread::create([prepare = std::move(prepare), progress, prepared, error]() {
		try {
			*prepared = prepare(*progress);
		} catch (const std::exception& e) {
			*error = plain(e.what());
		}
	});
	connect(_worker, &QThread::finished, this, [this, prepared, error]() {
		_poll_timer.stop();
		_worker->deleteLater();
		_worker = nullptr;
		_preparing.reset();
		_state = State::idle;
		if (not error->isEmpty()) {
			fail(tr("Cannot read the inputs"), *error);
		} else if (not prepared->has_value()) {
			next();
		} else {
			create(std::move(**prepared));
		}
	});
	_worker->start();
	_poll_timer.start();
}

void App::PVImportController::create(PVRush::PVSourceDescription description)
{
	const PVRush::PVFormat& format = description.get_format();
	const PVRush::PVInputType_p type = description.get_source_creator()->supported_type_lib();
	_name = type->tab_name_of_inputs(description.get_inputs());
	page().set_title(tr("Import of %1").arg(_name), type->icon());
	page().start_progress(tr("Preparing the import..."));

	const size_t axes_count = format.get_axes().size();
	if (axes_count < 2) {
		fail(tr("Format \"%1\" is invalid").arg(format.get_full_path()),
		     tr("It has %1, it must define at least 2 axes to be usable.")
		         .arg(axes_count == 0 ? tr("no axis") : tr("only one axis")));
		return;
	}

	Squey::PVRoot& root = _window.get_root();
	Squey::PVScene* scene = root.current_scene();
	if (scene == nullptr and not root.get_children().empty()) {
		scene = root.get_children().front();
	}
	if (scene == nullptr) {
		scene = &projects().project_new();
		_new_scene = scene;
		// Making a data collection selects its tab.
		projects().show_import_page();
	}
	root.select_scene(*scene);

	try {
		_source = &scene->emplace_add_child(description);
		// Before anything is read: it maps and scales each axis, which its type may not allow.
		_memory_forecast = std::make_unique<PVMemoryConsumptionForecast>(*_source);
	} catch (const Squey::InvalidScalingMapping& e) {
		fail(tr("Error with the format"),
		     tr("The mapping and the scaling of the types of the format are not compatible: %1")
		         .arg(plain(e.what())));
		return;
	} catch (const PVRush::PVFormatException& e) {
		fail(tr("Error with the format"), plain(e.what()));
		return;
	} catch (const PVRush::PVInputException& e) {
		fail(tr("Fatal error while loading source"), plain(e.what()));
		return;
	} catch (const std::exception& e) {
		fail(tr("Error while loading source"), plain(e.what()));
		return;
	}

	// A format the import detected, or made as pcap does, has no name.
	const QString format_name = _source->get_format_name();
	page().start_progress(
	    format_name.isEmpty() ? tr("Extracting...") : tr("Extracting %1...").arg(format_name),
	    {{tr("Discard"), [this]() { ask_discard(); }, PVModdedIcon("trash-xmark")},
	     {tr("Stop and process"), [this]() { stop_extraction(); }, PVModdedIcon("square-full")}});
	try {
		_job = _source->extract(0);
	} catch (const PVRush::PVInputException& e) {
		fail(tr("Cannot create sources"), tr("Error with input: %1").arg(plain(e.what())));
		return;
	} catch (const PVRush::PVNrawException& e) {
		fail(tr("Cannot create sources"), tr("Error with nraw: %1").arg(plain(e.what())));
		return;
	}
	_monitor_memory = true;
	_last_memory_check = std::chrono::steady_clock::now();
	_discarding = false;
	_state = State::extracting;
	_poll_timer.start();
}

void App::PVImportController::poll()
{
	if (_state == State::preparing) {
		page().set_progress(qint64(_preparing->value()), qint64(_preparing->maximum()));
		page().set_progress_details(_preparing->status());
		return;
	}
	if (_state == State::processing) {
		static const char* steps[] = {"Computing mapping...", "Computing scaling...",
		                              "Creating views...", "Computing parallel coordinates..."};
		const int step = _processing_step;
		page().set_progress(step, qint64(std::size(steps)));
		page().set_progress_details(step < int(std::size(steps)) ? tr(steps[step]) : QString());
		return;
	}
	if (_state != State::extracting) {
		return;
	}

	if (not _job->running()) {
		_poll_timer.stop();
		extraction_ended();
		return;
	}
	const QLocale locale(QLocale::English);
	page().set_progress_details(tr("Number of extracted events: %1\nNumber of rejected events: %2")
	                                .arg(locale.toString(qulonglong(_job->status())),
	                                     locale.toString(qulonglong(_job->rejected_elements()))));
	// A maximum takes the progress bar out of its undeterminate state.
	page().set_progress(qint64(_job->get_value()),
	                    _job->get_value() > 0 ? qint64(_source->max_size()) : 0);
	check_memory();
}

void App::PVImportController::check_memory()
{
	const auto now = std::chrono::steady_clock::now();
	if (not _monitor_memory or _question or now - _last_memory_check < MEMORY_CHECK_PERIOD) {
		return;
	}
	_last_memory_check = now;

	// A null value means the available memory could not be determined: skip the check rather
	// than read it as "no memory left".
	const size_t available_memory = PVCore::available_memory();
	if (available_memory == 0 or
	    (*_memory_forecast)(_source->get_row_count()) <=
	        available_memory / 100 *
	            size_t(PVGuiQt::settings::import_memory_warning_percent.value())) {
		return;
	}
	ask(tr("Continuing to import the data will likely exceed the available memory, and make the "
	       "system unstable. Stop the import here, and process the events extracted so far?"),
	    {{tr("Stop and process"), [this]() { stop_extraction(); }, PVModdedIcon("square-full")},
	     {tr("Continue"), [this]() { _monitor_memory = false; }, PVModdedIcon("play")}});
}

void App::PVImportController::ask(QString const& text,
                                  std::vector<PVGuiQt::PVImportPage::Action> actions)
{
	_job->pause(true);
	page().set_progress_buttons_enabled(false);
	for (PVGuiQt::PVImportPage::Action& action : actions) {
		action.on_click = [this, on_click = std::move(action.on_click)]() {
			_question = nullptr;
			page().set_progress_buttons_enabled(true);
			on_click();
			if (_job) {
				_job->pause(false);
			}
		};
	}
	_question =
	    page().add_message(PVGuiQt::PVImportPage::Severity::warning, text, std::move(actions));
}

void App::PVImportController::stop_extraction()
{
	if (_state != State::extracting) {
		return;
	}
	if (_question) {
		page().remove_message(_question);
		_question = nullptr;
		page().set_progress_buttons_enabled(true);
	}
	// What is extracted so far stays, for the processing.
	_job->cancel();
}

void App::PVImportController::ask_discard()
{
	if (_state != State::extracting or _question) {
		return;
	}
	ask(tr("Discard the events of %1 extracted so far? Nothing of it is kept.").arg(_name),
	    {{tr("Discard"),
	      [this]() {
		      _discarding = true;
		      _job->cancel();
	      },
	      PVModdedIcon("trash-xmark")},
	     {tr("Continue"), []() {}, PVModdedIcon("play")}});
}

void App::PVImportController::extraction_ended()
{
	_state = State::idle;
	if (_question) {
		page().remove_message(_question);
		_question = nullptr;
	}
	try {
		_source->wait_extract_end(_job);
	} catch (const PVRush::PVInputException& e) {
		fail(tr("Cannot create sources"), tr("Error with input: %1").arg(plain(e.what())));
		return;
	} catch (const PVRush::UnicodeSourceError&) {
		fail(tr("Cannot create sources"),
		     tr("The encoding of the files does not let Squey extract them."));
		return;
	} catch (const PVRush::PVConverterCreationError& e) {
		fail(tr("Unsupported charset"), plain(e.what()));
		return;
	} catch (const std::exception& e) {
		fail(tr("Fatal error while loading source"), plain(e.what()));
		return;
	}
	_job.reset();
	_memory_forecast.reset();

	if (_discarding) {
		drop_source();
		next();
		return;
	}

	if (_source->get_rushnraw().row_count() == 0) {
		const QString format_name = _source->get_format_name().toHtmlEscaped();
		QString text =
		    tr("<p>The files <strong>%1</strong> using format <strong>%2</strong> "
		       "cannot be opened. ")
		        .arg(QString::fromStdString(_source->get_name()).toHtmlEscaped(), format_name);
		const PVRow invalid_count = _source->get_invalid_evts().size();
		if (invalid_count > 0) {
			text += tr("Indeed, <strong>%1 elements</strong> have been extracted but were "
			           "<strong>all invalid</strong>.</p>"
			           "<p>This is because one or more splitters and/or filters defined in "
			           "format <strong>%2</strong> reported invalid events during the "
			           "extraction.<br />You may have invalid regular expressions set in this "
			           "format, or simply all the events have been invalidated by one or more "
			           "filters thus no events matches your criterias.</p>"
			           "<p>You might try to <strong>fix your format</strong> or try to load "
			           "<strong>another set of data</strong>.</p>")
			            .arg(invalid_count)
			            .arg(format_name);
		} else {
			text += tr("Indeed, the sources <strong>were empty</strong> (empty files, bad "
			           "database query, bad filtering parameters, etc...) because no elements "
			           "have been extracted.</p><p>You should try to load another set of "
			           "data.</p>");
		}
		fail(tr("Cannot load sources"), text);
		return;
	}

	process();
}

void App::PVImportController::process()
{
	_state = State::processing;
	_processing_step = 0;
	_processing_error.clear();
	page().start_progress(tr("Processing..."));

	Squey::PVSource* source = _source;
	_worker = QThread::create([this, source]() {
		try {
			auto& mapped = source->emplace_add_child();
			_processing_step = 1;
			auto& scaled = mapped.emplace_add_child();
			_processing_step = 2;
			Squey::PVView& view = scaled.emplace_add_child();
			_processing_step = 3;
			// The zone trees of its parallel coordinates, which their view would build as it
			// shows, under a dialog of its own.
			PVParallelView::common::get_rendering_context(view);
			_processing_step = 4;
		} catch (const std::exception& e) {
			_processing_error = plain(e.what());
		}
	});
	connect(_worker, &QThread::finished, this, &PVImportController::processed);
	_worker->start();
	_poll_timer.start();
}

void App::PVImportController::processed()
{
	_poll_timer.stop();
	_worker->deleteLater();
	_worker = nullptr;
	_state = State::idle;
	if (not _processing_error.isEmpty()) {
		fail(tr("Cannot process sources"), _processing_error);
		return;
	}

	Squey::PVSource& source = *_source;
	_source = nullptr;
	_new_scene = nullptr;
	_imported = true;
	_window.source_loaded(source, true);
	projects().show_errors_and_warnings();
	Q_EMIT _window.change_of_current_view_Signal();
	next();
}

void App::PVImportController::fail(QString const& title, QString const& text)
{
	_poll_timer.stop();
	_state = State::idle;
	drop_source();
	// A text of paragraphs starts on a line of its own already.
	page().add_message(PVGuiQt::PVImportPage::Severity::critical,
	                   QString("<b>%1: %2</b>%3")
	                       .arg(_name.toHtmlEscaped(), title.toHtmlEscaped(),
	                            text.startsWith("<p>") ? text : "<br>" + text));
	next();
}

void App::PVImportController::drop_source()
{
	_job.reset();
	_memory_forecast.reset();
	if (_source != nullptr) {
		_window.remove_source(_source);
		_source = nullptr;
	} else if (_new_scene != nullptr) {
		projects().remove_project(projects().get_workspace_tab_widget_from_scene(_new_scene));
	}
	_new_scene = nullptr;
}

void App::PVImportController::end()
{
	_busy = false;
	_new_scene = nullptr;
	page().end_progress();
	set_window_enabled(true);
	if (_content_import and _imported) {
		page().clear_content();
	}
	if (page().content() != nullptr) {
		// To import it again, or anew once the imports that were not its own are over.
		page().set_title(tr("Import: %1").arg(_content_type->menu_input_name()),
		                 _content_type->icon());
		page().set_content_shown(true);
		if (_content_import) {
			projects().show_import_page();
		}
	} else if (page().has_messages()) {
		page().set_title(tr("Import"));
		page().set_closable(true);
		projects().show_import_page();
	} else {
		projects().hide_import_page();
	}
	_content_import = false;
	Q_EMIT idle();
}

void App::PVImportController::set_window_enabled(bool enabled)
{
	// The shortcuts of actions in menus only work while the menu bar is enabled.
	_window.menuBar()->setEnabled(enabled);
	projects().set_importing(not enabled);
	if (not enabled) {
		for (QShortcut* shortcut : _window.findChildren<QShortcut*>()) {
			if (shortcut->isEnabled()) {
				shortcut->setEnabled(false);
				_disabled_shortcuts.emplace_back(shortcut);
			}
		}
	} else {
		for (const QPointer<QShortcut>& shortcut : _disabled_shortcuts) {
			if (shortcut) {
				shortcut->setEnabled(true);
			}
		}
		_disabled_shortcuts.clear();
	}
}
