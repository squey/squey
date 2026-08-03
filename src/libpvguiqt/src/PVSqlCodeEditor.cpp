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

#include <pvguiqt/PVSqlCodeEditor.h>

#include <pvkernel/core/PVTheme.h>

#include <squey/PVDuckDBQuery.h>

#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/definition.h>
#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/repository.h>
#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/syntaxhighlighter.h>
#include <KF6/KSyntaxHighlighting/KSyntaxHighlighting/theme.h>

#include <QAbstractItemView>
#include <QCompleter>
#include <QKeyEvent>
#include <QMimeData>
#include <QMouseEvent>
#include <QPair>
#include <QScrollBar>
#include <QStandardItemModel>
#include <QVector>

static const char* THEME_NAMES[] = {"ayu Light", "ayu Dark"};

// The tables a query can read, and what each stands for. "layers" first: it is
// what the listing shows, so it is the one a query is usually written about.
//
// The text forms come last: reading a cell the format could not parse is a
// deliberate act, not the everyday query.
static const QVector<QPair<QString, QString>> TABLES = {
    {"layers", "layers — every row the layer stack lets through"},
    {"selection", "selection — the current selection only"},
    {"layer('All events')", "layer('name') — one layer, by name"},
    {"layers(text := true)", "layers(text := true) — every column as written, unparsed cells included"},
    {"selection(text := true)", "selection(text := true) — the selection, as written"}};

// Keywords that expect something after them, and what that something is. Typing
// one of these opens the next list on its own.
static const QHash<QString, int> FOLLOWED_BY = {
    {"FROM", 1},  {"JOIN", 1},                                              // tables
    {"SELECT", 2}, {"WHERE", 2}, {"AND", 2},    {"OR", 2},   {"BY", 2},      // columns
    {"HAVING", 2}, {"ON", 2},    {"NOT", 2},    {"COUNT", 2}, {"SUM", 2},
    {"AVG", 2},    {"MIN", 2},   {"MAX", 2},    {"DISTINCT", 2}};

static const QStringList KEYWORDS = {
    "SELECT", "FROM",   "WHERE",  "AND",    "OR",     "NOT",   "IN",     "BETWEEN",
    "LIKE",   "ILIKE",  "IS",     "NULL",   "ORDER",  "BY",    "GROUP",  "HAVING",
    "LIMIT",  "OFFSET", "DISTINCT", "COUNT", "SUM",   "AVG",   "MIN",    "MAX",
    "AS",     "ASC",    "DESC",   "CASE",   "WHEN",   "THEN",  "ELSE",   "END",
    "WITH",   "UNION",  "EXCEPT", "INTERSECT"};

/**
 * Whether @a c belongs to the name under the cursor.
 *
 * Qt's WordUnderCursor stops at a dash, but axis names may hold one -- they are
 * exposed verbatim -- and a name that needs quoting carries its quotes as part
 * of what a query has to replace.
 */
static bool is_name_char(QChar c)
{
	return c.isLetterOrNumber() || c == '_' || c == '-' || c == '.' || c == '"';
}

/**
 * The name a token carries, without the double quotes it may need in order to
 * be written at all. The completer matches on the name itself, which is what
 * the user sees everywhere else in the application.
 */
static QString unquoted(QString token)
{
	if (token.startsWith('"')) {
		token.remove(0, 1);
	}
	if (token.endsWith('"')) {
		token.chop(1);
	}
	// A double quote inside a quoted identifier is written twice.
	return token.replace("\"\"", "\"");
}

PVGuiQt::PVSqlCodeEditor::PVSqlCodeEditor(QWidget* parent /* = nullptr */) : QTextEdit(parent)
{
	auto* repository = new KSyntaxHighlighting::Repository;
	auto* highlighter = new KSyntaxHighlighting::SyntaxHighlighter(document());

	QFont font = document()->defaultFont();
	font.setFamily("Monospace");
	font.setPointSizeF(10.5);
	font.setStyleStrategy(QFont::PreferAntialias);
	document()->setDefaultFont(font);

	highlighter->setDefinition(repository->definitionForName("SQL"));

	const auto& theme = repository->theme(THEME_NAMES[(size_t)PVCore::PVTheme::color_scheme()]);
	highlighter->setTheme(theme);
	setStyleSheet(
	    QString("QTextEdit { background-color : %1; }")
	        .arg(QColor(theme.editorColor(KSyntaxHighlighting::Theme::EditorColorRole::BackgroundColor))
	                 .name()));

	_model = new QStandardItemModel(this);
	_completer = new QCompleter(this);
	_completer->setWidget(this);
	_completer->setModel(_model);
	_completer->setCompletionMode(QCompleter::PopupCompletion);
	_completer->setCaseSensitivity(Qt::CaseInsensitive);
	// Match on the displayed line, which begins with the name: typing "por"
	// reaches "port — UINTEGER". What gets inserted is held apart, under
	// InsertRole, since it is not what is worth reading.
	_completer->setCompletionRole(Qt::DisplayRole);
	// By index rather than by string: the string a completion carries is its
	// description, and what has to be inserted is beside it.
	QObject::connect(_completer, QOverload<const QModelIndex&>::of(&QCompleter::activated), this,
	                 &PVSqlCodeEditor::insert_completion);
}

// What a business type is stored as, and how to get back and forth. A query
// reads the stored integer -- an address is a UINTEGER whose order is the
// address order -- so a column of one of these types is not readable, and not
// writable against, without a conversion.
//
// "write" comes first in the offered list: converting the literal leaves the
// comparison on the stored integer, where converting the column builds a string
// per row for the same answer.
namespace
{
struct axis_conversion {
	const char* axis_type;
	//! What the completer shows instead of the storage type.
	const char* shown_as;
	const char* write;
	const char* write_label;
	const char* read;
	const char* read_label;
};

const axis_conversion CONVERSIONS[] = {
    {"ipv4", "IPv4 address", "ipv4()", "ipv4('192.168.1.1') — an address, to compare a column against",
     "ipv4_text()", "ipv4_text(column) — the address a column holds, as text"},
    {"mac_address", "MAC address", "mac_address()",
     "mac_address('00:11:22:33:44:55') — an address, to compare a column against", "mac_address_text()",
     "mac_address_text(column) — the address a column holds, as text"},
    // A datetime is an epoch, and which epoch depends on the time format its
    // axis was given, so only the reading direction is offered -- and that one
    // is DuckDB's own.
    {"datetime", "date and time (epoch seconds)", nullptr, nullptr, "to_timestamp()",
     "to_timestamp(column) — the instant a column holds"},
    {"datetime_ms", "date and time (epoch milliseconds)", nullptr, nullptr, "epoch_ms()",
     "epoch_ms(column) — the instant a column holds"},
    {"ipv6", "IPv6 address", nullptr, nullptr, nullptr, nullptr},
};
} // namespace

void PVGuiQt::PVSqlCodeEditor::set_columns(const QStringList& names,
                                           const QStringList& types,
                                           const QStringList& axis_types)
{
	_column_names = names;
	_column_types = types;
	_column_axis_types = axis_types;

	// Only the conversions this source can use: offering ipv4() where no column
	// holds an address is one more thing to read past.
	_conversions.clear();
	for (const axis_conversion& conversion : CONVERSIONS) {
		if (not axis_types.contains(conversion.axis_type)) {
			continue;
		}
		if (conversion.write != nullptr) {
			_conversions.append(qMakePair(QString(conversion.write), QString(conversion.write_label)));
		}
		if (conversion.read != nullptr) {
			_conversions.append(qMakePair(QString(conversion.read), QString(conversion.read_label)));
		}
	}

	// The completer matches on the bare name -- that is what the user types --
	// while the text inserted is the quoted form when the name needs it.
	_quoted_form.clear();
	for (const QString& name : names) {
		const QString quoted =
		    QString::fromStdString(Squey::PVDuckDBQuery::quote_identifier(name.toStdString()));
		if (quoted != name) {
			_quoted_form.insert(name, quoted);
		}
	}
}

QString PVGuiQt::PVSqlCodeEditor::column_label(int index) const
{
	const QString& name = _column_names.at(index);
	const QString type = _column_types.value(index);
	if (type.isEmpty()) {
		return name;
	}

	// What Squey calls the axis, when SQL cannot show it as itself: an address
	// and a counter are both UINTEGER, and the type alone would not say which is
	// which -- nor that one of them has a conversion worth reaching for.
	const QString axis_type = _column_axis_types.value(index);
	for (const axis_conversion& conversion : CONVERSIONS) {
		if (axis_type == conversion.axis_type) {
			return QString("%1 — %2, stored as %3").arg(name, conversion.shown_as, type);
		}
	}
	return QString("%1 — %2").arg(name, type);
}

PVGuiQt::PVSqlCodeEditor::Context PVGuiQt::PVSqlCodeEditor::context_at_cursor() const
{
	// Walk back over the prefix being typed, then over blanks, then read the
	// word before it: that word is what says whether a table or a column is
	// expected here.
	const QString text = toPlainText();
	int pos = textCursor().position() - current_prefix().length();
	while (pos > 0 && text.at(pos - 1).isSpace()) {
		--pos;
	}
	// A comma or an opening parenthesis continues the previous clause.
	if (pos > 0 && (text.at(pos - 1) == ',' || text.at(pos - 1) == '(')) {
		--pos;
		while (pos > 0 && text.at(pos - 1).isSpace()) {
			--pos;
		}
	}

	int start = pos;
	while (start > 0 && text.at(start - 1).isLetterOrNumber()) {
		--start;
	}
	const QString word = text.mid(start, pos - start).toUpper();

	switch (FOLLOWED_BY.value(word, 0)) {
	case 1:
		return Context::Tables;
	case 2:
		return Context::Columns;
	default:
		return Context::Any;
	}
}

QString PVGuiQt::PVSqlCodeEditor::current_prefix() const
{
	// Walked back by hand rather than through WordUnderCursor, to the nearest
	// character that really separates SQL tokens. What it spans is also what an
	// accepted completion replaces, so the quotes of a quoted name are part of
	// it -- the inserted form carries its own.
	const QString text = toPlainText();
	const int end = textCursor().position();
	int start = end;
	while (start > 0 && is_name_char(text.at(start - 1))) {
		--start;
	}
	return text.mid(start, end - start);
}

void PVGuiQt::PVSqlCodeEditor::show_completions(bool force)
{
	// Matched on the bare name: a name that needs quoting is typed with its
	// quotes, but it is listed -- and therefore matched -- without them.
	show_completions(force, context_at_cursor(), unquoted(current_prefix()));
}

void PVGuiQt::PVSqlCodeEditor::show_completions(bool force, Context context, const QString& match)
{
	const QString prefix = current_prefix();

	if (not force && prefix.length() < 2) {
		_completer->popup()->hide();
		return;
	}

	// Rebuilt per keystroke so the offered set follows the context. The lists
	// are small enough that this costs nothing next to the popup itself.
	_model->clear();
	const auto add = [&](const QString& insert, const QString& label) {
		auto* item = new QStandardItem(label);
		item->setData(insert, InsertRole);
		_model->appendRow(item);
	};

	if (context == Context::Tables || context == Context::Any) {
		for (const auto& [table, label] : TABLES) {
			add(table, label);
		}
	}
	if (context == Context::Columns || context == Context::Any) {
		add("rowid", "rowid — physical row index");
		for (int i = 0; i < _column_names.size(); ++i) {
			const QString& name = _column_names.at(i);
			if (name == "rowid") {
				continue;
			}
			add(name, column_label(i));
		}
		// After the columns: they are what a position expects, and these are what
		// one wraps around them.
		for (const auto& [call, label] : _conversions) {
			add(call, label);
		}
	}
	if (context == Context::Any) {
		for (const QString& keyword : KEYWORDS) {
			add(keyword, keyword);
		}
	}

	_completer->setCompletionPrefix(match);
	if (_completer->completionCount() == 0) {
		_completer->popup()->hide();
		return;
	}
	_completer->popup()->setCurrentIndex(_completer->completionModel()->index(0, 0));

	QRect rect = cursorRect();
	rect.setWidth(_completer->popup()->sizeHintForColumn(0) +
	              _completer->popup()->verticalScrollBar()->sizeHint().width());
	_completer->complete(rect);
}

void PVGuiQt::PVSqlCodeEditor::insert_completion(const QModelIndex& index)
{
	const QString completion = index.data(InsertRole).toString();
	if (completion.isEmpty()) {
		return;
	}

	// Replace the whole prefix rather than appending the tail: the inserted
	// text may be quoted ("font-infos-family_name") while what the user typed
	// is not, so the two do not share a suffix.
	const QString inserted = _quoted_form.value(completion, completion);

	_inserting = true;
	QTextCursor cursor = textCursor();
	cursor.setPosition(cursor.position() - current_prefix().length(), QTextCursor::KeepAnchor);
	cursor.insertText(inserted);

	// A keyword that expects a follow-up gets its separating space and opens
	// the next list straight away, so a query is written by picking rather than
	// by remembering what comes next.
	const bool opens_next = FOLLOWED_BY.contains(completion.toUpper());
	if (opens_next) {
		cursor.insertText(" ");
	}
	// A conversion is offered as "ipv4()", and what comes next goes between the
	// parentheses rather than after them.
	if (inserted.endsWith("()")) {
		cursor.setPosition(cursor.position() - 1);
	}
	setTextCursor(cursor);
	_inserting = false;

	if (opens_next) {
		show_completions(true);
	}
}

QSize PVGuiQt::PVSqlCodeEditor::sizeHint() const
{
	const QFontMetrics metrics(document()->defaultFont());
	const int line = metrics.lineSpacing() + 2 * int(document()->documentMargin()) +
	                 2 * frameWidth();
	return {QTextEdit::sizeHint().width(), line};
}

QSize PVGuiQt::PVSqlCodeEditor::minimumSizeHint() const
{
	return {QTextEdit::minimumSizeHint().width(), sizeHint().height()};
}

void PVGuiQt::PVSqlCodeEditor::focusInEvent(QFocusEvent* event)
{
	_completer->setWidget(this);
	QTextEdit::focusInEvent(event);
}

bool PVGuiQt::PVSqlCodeEditor::complete_column_at_cursor()
{
	const QString text = toPlainText();
	const int position = textCursor().position();

	int start = position;
	while (start > 0 && is_name_char(text.at(start - 1))) {
		--start;
	}
	int end = position;
	while (end < text.size() && is_name_char(text.at(end))) {
		++end;
	}
	if (start == end || not _column_names.contains(unquoted(text.mid(start, end - start)))) {
		return false;
	}

	// The cursor goes to the end of the name first, because a completion
	// replaces what precedes it: clicked in the middle, it would leave the tail
	// of the old name behind.
	QTextCursor cursor = textCursor();
	cursor.setPosition(end);
	setTextCursor(cursor);

	// Columns rather than whatever the position would otherwise offer: the click
	// landed on one, so the answer to it is the others -- all of them, not the
	// ones that happen to be spelled like it. Reaching a different column is the
	// whole point, and the name already there says nothing about which.
	show_completions(true, Context::Columns, QString());
	return true;
}

void PVGuiQt::PVSqlCodeEditor::mouseReleaseEvent(QMouseEvent* event)
{
	QTextEdit::mouseReleaseEvent(event);

	// Reading a query is how one notices it names the wrong axis, so clicking
	// the name is the shortest way from noticing to fixing. Only on a name the
	// console knows: a click is also how a cursor is placed, and a popup on
	// every click would be in the way rather than at hand.
	if (event->button() != Qt::LeftButton || textCursor().hasSelection() ||
	    not complete_column_at_cursor()) {
		_completer->popup()->hide();
	}
}

void PVGuiQt::PVSqlCodeEditor::keyPressEvent(QKeyEvent* event)
{
	QAbstractItemView* popup = _completer->popup();

	// While the popup is up it owns navigation and validation, otherwise Enter
	// would insert a newline instead of accepting the highlighted entry.
	if (popup->isVisible()) {
		switch (event->key()) {
		case Qt::Key_Enter:
		case Qt::Key_Return:
		case Qt::Key_Tab:
		case Qt::Key_Escape:
		case Qt::Key_Up:
		case Qt::Key_Down:
			event->ignore();
			return;
		default:
			break;
		}
	}

	// Enter runs the query. A query is a line, so that is what Enter should do
	// with it; the line break moves to the modified keys, which is the bargain
	// every one-line query box makes.
	if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
		if ((event->modifiers() & (Qt::ControlModifier | Qt::ShiftModifier)) != 0) {
			insertPlainText("\n");
		} else {
			Q_EMIT run_requested();
		}
		return;
	}

	// Ctrl+Space asks for the list whatever has been typed, which is how an
	// empty position still offers what fits there.
	const bool forced =
	    event->key() == Qt::Key_Space && (event->modifiers() & Qt::ControlModifier) != 0;
	if (not forced) {
		QTextEdit::keyPressEvent(event);
		if (_inserting || event->text().isEmpty()) {
			return;
		}
	}

	show_completions(forced);
}

void PVGuiQt::PVSqlCodeEditor::insertFromMimeData(const QMimeData* source)
{
	QTextEdit::insertPlainText(source->text());
}
