/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <pvguiqt/PVDisplayViewSqlConsole.h>

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvguiqt/PVListDisplayDlg.h>
#include <pvguiqt/PVSqlCodeEditor.h>
#include <pvguiqt/PVSqlResultModel.h>
#include <pvguiqt/PVStatsModel.h>
#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVWorkspace.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/widgets/PVHelpWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <pvcop/db/algo.h>
#include <pvcop/db/array.h>

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <QGridLayout>
#include <QKeyEvent>
#include <QLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace
{

/**
 * Turn a value/count result into the model the distinct-values listing uses.
 *
 * Rather than re-render a count column, the result is handed to PVStatsModel:
 * the bars, the log scale, the count/percentage switches and the context menu
 * are then the very same code, not a lookalike.
 */
PVGuiQt::PVStatsModel* stats_model_from(const Squey::PVDuckDBQuery::Table& table)
{
	std::vector<std::string> values;
	values.reserve(table.rows.size());
	for (const auto& row : table.rows) {
		values.emplace_back(row[0]);
	}

	// The values column carries its own dictionary, since it holds query output
	// rather than a column of the source.
	pvcop::db::array col1 = pvcop::db::make_string_array(values);

	pvcop::db::array col2("number_uint64", table.rows.size());
	uint64_t total = 0;
	if (not table.rows.empty()) {
		auto& counts = col2.to_core_array<uint64_t>();
		for (size_t i = 0; i < table.rows.size(); ++i) {
			counts[i] = std::strtoull(table.rows[i][1].c_str(), nullptr, 10);
			total += counts[i];
		}
	}

	// The absolute reference is the total of the counts: each bar then reads as
	// a share of everything the query grouped over.
	pvcop::db::array abs_max("number_uint64", 1);
	abs_max.to_core_array<uint64_t>()[0] = total;

	pvcop::db::array minmax = pvcop::db::algo::minmax(col2);

	return new PVGuiQt::PVStatsModel(
	    QString::fromStdString(table.column_names[1]),
	    QString::fromStdString(table.column_names[0]), QString(), std::move(col1),
	    std::move(col2), std::move(abs_max), std::move(minmax));
}

/**
 * Show a result listing as a dock of the workspace, next to the other listings.
 *
 * A result is read against the views it was written to explain, so it belongs
 * beside them rather than in a window floating over them -- which is how the
 * distinct-values listing is shown, and this is the same route: the dialog is
 * handed to the workspace, which tabifies it with whatever already occupies
 * that area.
 *
 * The console can live outside a workspace (a test harness builds it alone),
 * and a result is worth seeing there too, so it then falls back to a window of
 * its own.
 */
void dock_result(const PVDisplays::PVDisplayViewIf& display,
                 Squey::PVView* view,
                 QWidget* console_widget,
                 PVGuiQt::PVListDisplayDlg* dlg)
{
	auto* workspace = PVCore::get_qobject_parent_of_type<PVGuiQt::PVWorkspaceBase*>(console_widget);
	if (workspace == nullptr) {
		dlg->show();
		return;
	}

	// The dialog's own Close button would sit inside the dock, which already has
	// one in its title bar.
	delete dlg->findChild<QWidget*>("buttonBox");

	// The dock reads the display for its flags and its status-bar signals. Only
	// create_widget() is const -- the registry holds this display by non-const
	// reference and hands it to every other entry point that way.
	workspace->add_view_display(view, dlg, const_cast<PVDisplays::PVDisplayViewIf&>(display), true,
	                            Qt::RightDockWidgetArea);
}

/**
 * The console's widget.
 *
 * It is a class of its own only to answer the help key: the dock's "?" button
 * posts one, and every display carrying a help page catches it this way.
 */
class console_widget : public QWidget
{
  public:
	using QWidget::QWidget;

  protected:
	void keyPressEvent(QKeyEvent* event) override
	{
		if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
			show_help();
			return;
		}
		QWidget::keyPressEvent(event);
	}

	bool eventFilter(QObject* watched, QEvent* event) override
	{
		if (watched == _help && event->type() == QEvent::Hide) {
			close_help();
		}
		return QWidget::eventFilter(watched, event);
	}

  public:
	/**
	 * Put @a error under the console, or take the line away when it is empty.
	 *
	 * A dock keeps whatever height it was given, so the room the message needed
	 * stays taken once the message is gone: the console would creep up a line
	 * at every mistake and never come back down. The height it had before the
	 * first error is therefore kept, and given back once the errors stop.
	 */
	void set_error(QLabel* label, const QString& error)
	{
		const bool had = label->isVisible();
		const bool has = not error.isEmpty();

		auto* dock = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(this);
		// Read before the line is put up: showing it is what makes the dock
		// grow, so afterwards there is no height left to remember.
		if (has && not had && dock != nullptr) {
			_height_without_error = dock->height();
		}

		label->setText(error);
		label->setVisible(has);

		if (has || not had || _height_without_error <= 0) {
			return;
		}

		const int height = _height_without_error;
		_height_without_error = 0;
		// Deferred, and only after the layout has been told to recompute: a
		// dock cannot be resized below what its contents still demand, and the
		// widget goes on demanding room for the line until its layout has been
		// activated with the line hidden.
		QTimer::singleShot(0, this, [this, height]() {
			if (layout() != nullptr) {
				layout()->activate();
			}
			updateGeometry();
			auto* workspace =
			    PVCore::get_qobject_parent_of_type<PVGuiQt::PVWorkspaceBase*>(this);
			auto* target = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(this);
			if (workspace != nullptr && target != nullptr) {
				workspace->resizeDocks({target}, {height}, Qt::Vertical);
			}
		});
	}

  private:
	void show_help()
	{
		// Over this widget, the way every other view's help covers the view it
		// belongs to. Built on first use because the dock it will be measured
		// against does not exist yet when the console is created.
		if (_help == nullptr) {
			_help = new PVWidgets::PVHelpWidget(this);
			_help->hide();
			// The reference first, the worked example last and across the width:
			// it puts together what the blocks above name, and read before them it
			// would be a query with nothing to hang on.
			_help->initTextFromFile("SQL console's help");
			_help->addTextFromFile(":help-sql-console-tables");
			_help->newColumn();
			_help->addTextFromFile(":help-sql-console-writing");
			_help->newTable();
			_help->addTextFromFile(":help-sql-console-types");
			_help->newColumn();
			_help->addTextFromFile(":help-sql-console-shortcuts");
			_help->newTable();
			_help->addTextFromFile(":help-sql-console-completion");
			_help->newTable();
			_help->addTextFromFile(":help-sql-console-example");
			_help->finalizeText();
			_help->installEventFilter(this);
		}
		if (not _help->isHidden()) {
			return;
		}

		_help->popup(this, PVWidgets::PVTextPopupWidget::AlignCenter,
		             PVWidgets::PVTextPopupWidget::ExpandAll);

		// The console is one line tall, which is no room for a page of text, so
		// the dock opens for as long as the page is up. Done after the popup
		// rather than before: the help follows its parent's resizes on its own,
		// so this is the one call that has to happen, in either order.
		auto* workspace = PVCore::get_qobject_parent_of_type<PVGuiQt::PVWorkspaceBase*>(this);
		auto* dock = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(this);
		if (workspace == nullptr || dock == nullptr) {
			return;
		}
		_closed_height = dock->height();
		workspace->resizeDocks({dock}, {std::max(320, workspace->height() / 2)}, Qt::Vertical);
	}

	//! Give the dock back the height it had before the page went up.
	void close_help()
	{
		if (_closed_height <= 0) {
			return;
		}
		auto* workspace = PVCore::get_qobject_parent_of_type<PVGuiQt::PVWorkspaceBase*>(this);
		auto* dock = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(this);
		if (workspace != nullptr && dock != nullptr) {
			workspace->resizeDocks({dock}, {_closed_height}, Qt::Vertical);
		}
		_closed_height = 0;
	}

  private:
	PVWidgets::PVHelpWidget* _help = nullptr;
	//! The dock's height with the page down, kept while it is up. 0 when it is.
	int _closed_height = 0;
	//! Its height before an error took room. 0 while no error is shown.
	int _height_without_error = 0;
};

} // namespace

PVDisplays::PVDisplayViewSqlConsole::PVDisplayViewSqlConsole()
    // At the bottom, across the whole width: a query is written about everything
    // above it, and it is a line of text rather than a picture -- wide and short
    // is the shape it wants, and it is the one place that does not take room from
    // the views.
    // One per view, not one per press: a console holds the query being written,
    // so its toolbar button is a state -- shown or not -- rather than a command
    // that stacks another empty one beside it.
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget |
                          PVDisplayIf::UniquePerParameters | PVDisplayIf::HasHelpPage,
                      "SQL console",
                      PVModdedIcon("database-play"),
                      Qt::BottomDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewSqlConsole::create_widget(Squey::PVView* view,
                                                            QWidget* parent,
                                                            Params const&) const
{
	auto& source = view->get_parent<Squey::PVSource>();

	auto* console_widget = new class console_widget(parent);

	// One query object per console: it holds its own DuckDB instance, and
	// building it is cheap since no data is copied into it. Bound to the view
	// rather than to the source, so "selection" and the layers a query names are
	// the ones this console sits next to -- a source may carry several views.
	auto query = std::make_shared<Squey::PVDuckDBQuery>(*view);

	// Column names feed the completer: they are the axis names verbatim, so
	// Ctrl+Space is how the schema is discovered rather than a printed list,
	// which would not fit a source with many axes.
	QStringList columns;
	for (const std::string& name : query->column_names()) {
		columns << QString::fromStdString(name);
	}
	QStringList types;
	for (const std::string& type : query->column_types()) {
		types << QString::fromStdString(type);
	}
	// What Squey calls each axis, alongside what SQL calls it: an address is
	// exposed as an integer, and only this says so.
	QStringList axis_types;
	for (const std::string& type : query->column_axis_types()) {
		axis_types << QString::fromStdString(type);
	}

	auto* editor = new PVGuiQt::PVSqlCodeEditor(console_widget);
	editor->set_columns(columns, types, axis_types);

	// And the other sources of the project, which a query can name but nothing
	// in the console otherwise shows: the completer is where they are found.
	QVector<PVGuiQt::PVSqlCodeEditor::SourceCompletion> sources;
	for (const Squey::PVDuckDBQuery::SourceInfo& info : query->sources()) {
		PVGuiQt::PVSqlCodeEditor::SourceCompletion described;
		described.name = QString::fromStdString(info.name);
		described.position = int(info.position);
		described.has_schema = info.has_schema;
		described.current = info.current;
		for (const std::string& name : info.column_names) {
			described.column_names << QString::fromStdString(name);
		}
		for (const std::string& type : info.column_types) {
			described.column_types << QString::fromStdString(type);
		}
		sources.append(described);
	}
	editor->set_sources(sources);

	// Asked per list rather than captured: layers are created, renamed and
	// dropped while the console stays open.
	editor->set_layer_provider([query]() {
		QStringList names;
		for (const std::string& name : query->layer_names()) {
			names << QString::fromStdString(name);
		}
		return names;
	});

	// The form worth teaching, said rather than written: a seeded query has to
	// be cleared before anything else can be typed, and one that is run as it
	// stands answers about a column nobody asked about. The hint carries the
	// same lesson -- SQL quoting being the reverse of most languages, a quoted
	// literal in front of the user is the thing worth showing -- and it costs
	// no keystrokes to be rid of.
	editor->setPlaceholderText(
	    "A condition, e.g. port = 80 AND host LIKE '%.fr' — or a full SELECT rowid FROM ...");

	// Only errors are shown, and only while there is one: a row count and a
	// duration are what the rest of the window already tells, and a line that is
	// almost always saying nothing takes room from the query.
	auto* status = new QLabel(console_widget);
	status->setWordWrap(true);
	status->setTextInteractionFlags(Qt::TextSelectableByMouse);
	status->setStyleSheet("QLabel { color : red; }");
	status->hide();

	// An icon rather than a label: the console is a strip, and Enter runs the
	// query anyway -- the button is there for the hand that reaches for it.
	auto* run_button = new QPushButton(console_widget);
	run_button->setIcon(PVModdedIcon("play"));
	run_button->setToolTip("Run the query (Enter)");
	run_button->setDefault(true);
	run_button->setFlat(true);
	run_button->setFixedSize(28, 28);
	run_button->setIconSize(QSize(14, 14));

	QObject::connect(run_button, &QPushButton::clicked, [=, this, &source]() {
		const std::string sql = editor->toPlainText().toStdString();
		if (sql.find_first_not_of(" \t\r\n") == std::string::npos) {
			return;
		}

		// Squey::PVSelection rather than its PVSelBitField base: that is what
		// the view takes, and the query only needs the base to fill it.
		Squey::PVSelection result(source.get_row_count());
		Squey::PVDuckDBQuery::Table table;
		QString error;

		// A query projecting rowid becomes a selection; anything else -- a
		// group-by, a count -- is displayed instead of being rejected, since
		// that is precisely what one writes to understand a dataset.
		const bool as_selection = query->yields_selection(sql);
		// A bare predicate reads the current selection, which is how every other
		// filter in the application composes: what one asks for narrows what one
		// already has. A query naming its own table -- layers, layer('name') --
		// says so and reaches wider.
		const PVCore::PVSelBitField* input = &view->get_real_output_selection();

		// The query runs in the progress box's worker so a long scan does not
		// freeze the window; it only touches the nraw and its own DuckDB
		// instance, and the view is updated afterwards, back on this thread.
		PVCore::PVProgressBox::progress(
		    [&](PVCore::PVProgressBox&) {
			    try {
				    if (as_selection) {
					    query->select(sql, *input, result);
				    } else {
					    table = query->run_tabular(sql, input);
				    }
			    } catch (const std::exception& e) {
				    error = QString::fromUtf8(e.what());
			    }
		    },
		    "Running SQL query...", console_widget);

		console_widget->set_error(status, error);
		if (not error.isEmpty()) {
			return;
		}

		if (not as_selection) {
			// A value/count pair is exactly what the distinct-values listing
			// shows, so it goes through the same dialog -- bars, scales and
			// context menu included. It has to be PVAbstractListStatsDlg rather
			// than its PVListDisplayDlg base: the count column is drawn by
			// PVListStringsDelegate, which only that dialog installs, and
			// PVStatsModel returns nothing for it on DisplayRole. Anything
			// wider than two columns is a plain grid.
			PVGuiQt::PVListDisplayDlg* dlg = nullptr;
			if (table.is_value_count()) {
				// Rebuilt from the query rather than captured, so the listing
				// follows the selection: the dialog calls this back whenever it
				// changes, which re-runs the query against the new one.
				auto create_model = [query, sql](const Squey::PVView&, PVCol,
				                                 const Squey::PVSelection& sel)
				    -> PVGuiQt::PVStatsModel* {
					return stats_model_from(query->run_tabular(sql, &sel));
				};
				auto* stats_dlg = new PVGuiQt::PVAbstractListStatsDlg(
				    *view, PVCol(0), create_model, true, console_widget);
				// A query groups over an expression of its own choosing, not
				// over an axis, so there is no column to search these values
				// back into: acting on a row would select rows unrelated to
				// what is displayed. This also makes the PVCol(0) above inert,
				// as nothing else reads it.
				stats_dlg->disable_selection_actions();
				dlg = stats_dlg;
			} else {
				dlg = new PVGuiQt::PVListDisplayDlg(new PVGuiQt::PVSqlResultModel(std::move(table)),
				                                    console_widget);
			}
			dlg->setAttribute(Qt::WA_DeleteOnClose);
			// Results stack as tabs of the same dock area, so the query itself is
			// the title: "SQL result" repeated tells one tab from another only by
			// its position. Truncated, since a tab label is a few words wide.
			QString title = editor->toPlainText().simplified();
			if (title.size() > 40) {
				title = title.left(39) + QChar(0x2026); // ellipsis
			}
			dlg->setWindowTitle(title);
			dock_result(*this, view, console_widget, dlg);
			return;
		}

		view->set_selection_view(result);
	});

	// Enter runs the query, so the button is a second way rather than the way.
	QObject::connect(editor, &PVGuiQt::PVSqlCodeEditor::run_requested, run_button,
	                 &QPushButton::click);

	// Beside the editor rather than over it, and anchored to the bottom: on a
	// one-line console the two read as the same corner, and on a console pulled
	// open the button stays where it was rather than riding up with the text.
	auto* layout = new QGridLayout;
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(editor, 0, 0);
	layout->addWidget(run_button, 0, 1, Qt::AlignBottom);
	layout->addWidget(status, 1, 0, 1, 2);
	layout->setRowStretch(0, 1);
	layout->setColumnStretch(0, 1);

	console_widget->setLayout(layout);
	console_widget->setWindowTitle(default_window_title(*view));
	return console_widget;
}
