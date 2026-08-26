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
#include <QListView>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPair>
#include <QFrame>
#include <QScrollBar>
#include <QStandardItemModel>
#include <QStyledItemDelegate>
#include <QVector>

static const char* THEME_NAMES[] = {"ayu Light", "ayu Dark"};

namespace
{

//! What the popup is painted with, per theme. Everything else derives from these.
struct popup_palette {
	QColor background;
	QColor border;
	QColor selection;
	QColor name;
	QColor detail;
};

popup_palette palette_of_theme()
{
	if (PVCore::PVTheme::is_color_scheme_dark()) {
		return {QColor(0x1f, 0x24, 0x30), QColor(0x39, 0x41, 0x50),
		        QColor(0x39, 0x8b, 0xf5, 0x59), QColor(0xe6, 0xe9, 0xef),
		        QColor(0x8b, 0x94, 0xa6)};
	}
	return {QColor(0xff, 0xff, 0xff), QColor(0xd4, 0xd9, 0xe1),
	        QColor(0x39, 0x8b, 0xf5, 0x33), QColor(0x1c, 0x21, 0x2b),
	        QColor(0x78, 0x82, 0x94)};
}

/**
 * Draws a completion as a name and, dimmed beside it, what that name holds.
 *
 * A style sheet cannot do this: the two halves need different colours within
 * one line, and the row needs a rounded highlight rather than the square band a
 * view paints. Painted rather than laid out through QTextDocument, which would
 * parse markup per row for a result no richer than two runs of text.
 */
class completion_delegate : public QStyledItemDelegate
{
  public:
	using QStyledItemDelegate::QStyledItemDelegate;

	static constexpr int RADIUS = 6;
	static constexpr int PADDING_X = 10;
	static constexpr int PADDING_Y = 5;
	static constexpr int GAP = 12;

	void paint(QPainter* painter,
	           const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override
	{
		const popup_palette colors = palette_of_theme();
		painter->save();
		painter->setRenderHint(QPainter::Antialiasing, true);

		// Inset, so that consecutive highlights read as separate pills rather
		// than as one band split by a hairline.
		const QRectF row = QRectF(option.rect).adjusted(3, 1, -3, -1);
		if (option.state & QStyle::State_Selected) {
			QPainterPath path;
			path.addRoundedRect(row, RADIUS, RADIUS);
			painter->fillPath(path, colors.selection);
		}

		const QString name = index.data(PVGuiQt::PVSqlCodeEditor::NameRole).toString();
		const QString detail = index.data(PVGuiQt::PVSqlCodeEditor::DetailRole).toString();

		QFont name_font = option.font;
		name_font.setBold(true);
		painter->setFont(name_font);
		painter->setPen(colors.name);
		const QFontMetrics name_metrics(name_font);
		const int text_top = option.rect.top();
		const int height = option.rect.height();
		painter->drawText(
		    QRect(option.rect.left() + PADDING_X, text_top,
		          option.rect.width() - 2 * PADDING_X, height),
		    Qt::AlignVCenter | Qt::AlignLeft, name);

		if (not detail.isEmpty()) {
			QFont detail_font = option.font;
			detail_font.setBold(false);
			painter->setFont(detail_font);
			painter->setPen(colors.detail);
			const int offset = name_metrics.horizontalAdvance(name) + GAP;
			const QRect area(option.rect.left() + PADDING_X + offset, text_top,
			                 option.rect.width() - 2 * PADDING_X - offset, height);
			if (area.width() > 0) {
				const QString elided =
				    QFontMetrics(detail_font).elidedText(detail, Qt::ElideRight, area.width());
				painter->drawText(area, Qt::AlignVCenter | Qt::AlignLeft, elided);
			}
		}

		painter->restore();
	}

	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override
	{
		const QString name = index.data(PVGuiQt::PVSqlCodeEditor::NameRole).toString();
		const QString detail = index.data(PVGuiQt::PVSqlCodeEditor::DetailRole).toString();

		QFont name_font = option.font;
		name_font.setBold(true);
		const QFontMetrics name_metrics(name_font);
		const QFontMetrics detail_metrics(option.font);

		int width = 2 * PADDING_X + name_metrics.horizontalAdvance(name);
		if (not detail.isEmpty()) {
			width += GAP + detail_metrics.horizontalAdvance(detail);
		}
		return QSize(width, name_metrics.height() + 2 * PADDING_Y);
	}
};

/**
 * The completion list, painting its own rounded background.
 *
 * A translucent window is what makes the corners actually round -- an opaque
 * one repaints the square the radius cut away -- but translucency also means
 * nothing fills the widget any more: neither the style sheet's background-color
 * nor the viewport, which has to stay unfilled for the corners to show through.
 * So the background is drawn here, under the items, and the style sheet is left
 * with the scroll bar alone.
 */
class completion_popup : public QListView
{
  public:
	using QListView::QListView;

	static constexpr int RADIUS = 8;

  protected:
	void paintEvent(QPaintEvent* event) override
	{
		const popup_palette colors = palette_of_theme();
		{
			QPainter painter(viewport());
			painter.setRenderHint(QPainter::Antialiasing, true);
			// Half a pixel in, so the one-pixel border lands inside the widget
			// rather than straddling its edge and coming out blurred.
			const QRectF area = QRectF(viewport()->rect()).adjusted(0.5, 0.5, -0.5, -0.5);
			QPainterPath path;
			path.addRoundedRect(area, RADIUS, RADIUS);
			painter.fillPath(path, colors.background);
			painter.setPen(QPen(colors.border, 1));
			painter.drawPath(path);
		}
		// The items go on top: QListView opens its own painter on the same
		// viewport, which is why the one above is closed first.
		QListView::paintEvent(event);
	}
};

/**
 * How a query writes a scope of @a source.
 *
 * The schema form where it exists, since that is the one worth teaching, and
 * the argument form otherwise -- a namesake has no schema, and its position is
 * the only thing telling it from the others.
 */
QString source_reference(const PVGuiQt::PVSqlCodeEditor::SourceCompletion& source,
                         const QString& scope = QStringLiteral("selection"))
{
	if (source.has_schema) {
		return QString::fromStdString(
		           Squey::PVDuckDBQuery::quote_identifier(source.name.toStdString())) +
		       "." + scope;
	}
	QString call = scope + "(source := '" + QString(source.name).replace("'", "''") + "'";
	if (source.position != 0) {
		call += ", source_position := " + QString::number(source.position);
	}
	return call + ")";
}

} // namespace


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

	// Set before the delegate: setPopup() drops the one the previous view held.
	_completer->setPopup(new completion_popup);
	_completer->popup()->setItemDelegate(new completion_delegate(_completer->popup()));
	restyle_popup();
	// The scheme can change while the console is open, and a popup painted for
	// the other one would be unreadable rather than merely out of place.
	QObject::connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this,
	                 [this]() { restyle_popup(); });
}

/**
 * The popup is a window of its own, so the console's own style sheet does not
 * reach it and its corners are the ones the platform draws.
 *
 * Rounding them takes both halves: a translucent frameless window, so that what
 * falls outside the radius is not painted over by an opaque square, and a style
 * sheet drawing the rounded background inside it.
 */
void PVGuiQt::PVSqlCodeEditor::restyle_popup()
{
	QAbstractItemView* popup = _completer->popup();
	const popup_palette colors = palette_of_theme();

	popup->setWindowFlags(popup->windowFlags() | Qt::FramelessWindowHint |
	                      Qt::NoDropShadowWindowHint);
	popup->setAttribute(Qt::WA_TranslucentBackground);
	// Both have to stay unfilled for the corners to show through; what would
	// otherwise be left is a bare window, so completion_popup paints the
	// rounded background itself.
	popup->viewport()->setAutoFillBackground(false);
	popup->viewport()->setAttribute(Qt::WA_TranslucentBackground);
	popup->setFrameShape(QFrame::NoFrame);
	popup->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	if (auto* list = qobject_cast<QListView*>(popup)) {
		// The delegate paints the highlight itself, as a pill; the view's own
		// alternating and hover bands would show through underneath.
		list->setAlternatingRowColors(false);
		list->setUniformItemSizes(false);
		list->setSpacing(0);
	}

	// Room for the painted border and radius, which the items must not sit on.
	popup->setContentsMargins(completion_popup::RADIUS / 2, completion_popup::RADIUS / 2,
	                          completion_popup::RADIUS / 2, completion_popup::RADIUS / 2);

	// No background here: it is painted, and a colour set through the style
	// sheet would be a square one laid outside the radius.
	popup->setStyleSheet(
	    QString("QAbstractItemView { background: transparent; border: none; outline: none; }"
	            "QAbstractItemView::item { border: none; }"
	            "QScrollBar:vertical {"
	            "  background: transparent; width: 8px; margin: 6px 3px 6px 0;"
	            "}"
	            "QScrollBar::handle:vertical {"
	            "  background: %1; border-radius: 4px; min-height: 24px;"
	            "}"
	            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
	            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
	            "  background: transparent;"
	            "}")
	        .arg(colors.detail.name(QColor::HexArgb)));
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

void PVGuiQt::PVSqlCodeEditor::set_sources(const QVector<SourceCompletion>& sources)
{
	_sources = sources;

	// Their columns are inserted the same way the console's own are: bare in
	// the popup, quoted in the query where the name needs it.
	for (const SourceCompletion& source : _sources) {
		if (source.current) {
			continue;
		}
		for (const QString& name : source.column_names) {
			const QString quoted =
			    QString::fromStdString(Squey::PVDuckDBQuery::quote_identifier(name.toStdString()));
			if (quoted != name) {
				_quoted_form.insert(name, quoted);
			}
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
	// The display role stays the whole line: it is what the completer filters
	// on, so typing "por" has to reach "port — UINTEGER". The halves are held
	// beside it for the delegate, which paints them differently.
	const auto add = [&](const QString& insert, const QString& label) {
		auto* item = new QStandardItem(label);
		item->setData(insert, InsertRole);
		// Split on the em dash the labels are built with; a label without one
		// is a name on its own, like a keyword.
		const int dash = label.indexOf(" — ");
		item->setData(dash < 0 ? label : label.left(dash), NameRole);
		item->setData(dash < 0 ? QString() : label.mid(dash + 3), DetailRole);
		_model->appendRow(item);
	};

	if (context == Context::Tables || context == Context::Any) {
		for (const auto& [table, label] : TABLES) {
			add(table, label);
		}
		// The scopes of the other sources, which a query can only name once it
		// knows they are there. The short form first where it exists, since it
		// is the one worth writing.
		for (const SourceCompletion& source : _sources) {
			if (source.current) {
				continue;
			}
			const QString from = source_reference(source);
			add(from, from + " — " + tr("the selection of %1").arg(source.name));
			add(source_reference(source, "layers"),
			    source_reference(source, "layers") + " — " +
			        tr("every row %1 lets through").arg(source.name));
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
		// Then the columns of the other sources. What gets inserted is the bare
		// name: in a join it is written against an alias, which only the query
		// knows -- so the source is said in the description rather than guessed
		// at in the insertion.
		for (const SourceCompletion& source : _sources) {
			if (source.current) {
				continue;
			}
			for (int i = 0; i < source.column_names.size(); ++i) {
				const QString& name = source.column_names.at(i);
				if (name == "rowid") {
					continue;
				}
				const QString type = source.column_types.value(i);
				add(name,
				    name + " — " + (type.isEmpty() ? source.name : type + ", " + source.name));
			}
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
	// sizeHintForColumn() measures the items alone: the padding and border the
	// style sheet draws around them are not in it, and a popup sized without
	// them elides the very description it exists to show.
	const QMargins frame = _completer->popup()->contentsMargins();
	rect.setWidth(_completer->popup()->sizeHintForColumn(0) + frame.left() + frame.right() +
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
