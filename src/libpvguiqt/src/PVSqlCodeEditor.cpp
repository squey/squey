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

#include <algorithm>

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
#include <QTimer>
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

		if (index.data(PVGuiQt::PVSqlCodeEditor::SectionRole).toBool()) {
			QFont heading_font = option.font;
			heading_font.setBold(true);
			heading_font.setCapitalization(QFont::AllUppercase);
			heading_font.setPointSizeF(std::max(6.5, option.font.pointSizeF() - 1.5));
			heading_font.setLetterSpacing(QFont::PercentageSpacing, 108);
			painter->setFont(heading_font);
			painter->setPen(colors.detail);
			painter->drawText(
			    QRect(option.rect.left() + PADDING_X, option.rect.top(),
			          option.rect.width() - 2 * PADDING_X, option.rect.height()),
			    Qt::AlignVCenter | Qt::AlignLeft,
			    index.data(PVGuiQt::PVSqlCodeEditor::NameRole).toString());
			painter->restore();
			return;
		}

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
		if (index.data(PVGuiQt::PVSqlCodeEditor::SectionRole).toBool()) {
			QFont heading_font = option.font;
			heading_font.setBold(true);
			heading_font.setCapitalization(QFont::AllUppercase);
			heading_font.setPointSizeF(std::max(6.5, option.font.pointSizeF() - 1.5));
			const QFontMetrics metrics(heading_font);
			// Taller than the text: a heading needs air above it to read as the
			// start of a group rather than as another row.
			return QSize(2 * PADDING_X + metrics.horizontalAdvance(name),
			             metrics.height() + 3 * PADDING_Y);
		}
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
	static constexpr int RADIUS = 8;
	//! Width of the floating bar, and how far its right edge sits from the border.
	static constexpr int BAR_WIDTH = 8;
	static constexpr int BAR_INSET = 4;

	completion_popup()
	{
		// The bar floats over the list rather than beside it. A scroll area lays
		// its bar out in a strip taken from the widget's edge, which is exactly
		// where the rounded corners are -- and the background is painted on the
		// viewport, which that strip is outside of, so the bar came out drawn on
		// the transparent pixels beyond the radius.
		//
		// With the view's own bar switched off, no strip is taken, the viewport
		// spans the whole widget and the background reaches every corner. This
		// one is a child of the view, placed by hand, and mirrors the bar that
		// is no longer shown.
		setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

		_bar = new QScrollBar(Qt::Vertical, this);
		_bar->hide();
		// On the bar itself rather than through the popup's sheet: the theme
		// styles scroll bar handles too, and set through an ancestor ours lost
		// to it -- the handle came out square and inset by a pixel. Set here it
		// wins, whatever the documented precedence says.
		style_bar();
		connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this,
		        &completion_popup::style_bar);

		QScrollBar* real = verticalScrollBar();
		connect(real, &QScrollBar::rangeChanged, this, [this](int low, int high) {
			_bar->setRange(low, high);
			_bar->setVisible(high > low);
			place_bar();
		});
		connect(real, &QScrollBar::valueChanged, _bar, &QScrollBar::setValue);
		connect(_bar, &QScrollBar::valueChanged, real, &QScrollBar::setValue);
		_bar->setPageStep(real->pageStep());
		connect(real, &QScrollBar::actionTriggered, this,
		        [this, real](int) { _bar->setPageStep(real->pageStep()); });
	}

  protected:
	void resizeEvent(QResizeEvent* event) override
	{
		QListView::resizeEvent(event);
		_bar->setPageStep(verticalScrollBar()->pageStep());
		place_bar();
	}

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

  private:
	//! A pill of the popup's dim colour, with nothing the theme drew left on it.
	void style_bar()
	{
		_bar->setStyleSheet(
		    QString("QScrollBar:vertical {"
		            "  background: transparent; border: none; width: %2px; margin: 0;"
		            "}"
		            "QScrollBar::handle:vertical, QScrollBar::handle:vertical:hover {"
		            "  background: %1; border: none; border-radius: %3px; min-height: 24px;"
		            "}"
		            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
		            "  height: 0; border: none; background: transparent;"
		            "}"
		            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
		            "  background: transparent; border: none;"
		            "}")
		        .arg(palette_of_theme().detail.name(QColor::HexArgb))
		        .arg(BAR_WIDTH)
		        .arg(BAR_WIDTH / 2));
	}

	//! Down the right edge, inside the radius, clear of the rounded corners.
	void place_bar()
	{
		_bar->setGeometry(width() - BAR_WIDTH - BAR_INSET, RADIUS, BAR_WIDTH,
		                  std::max(0, height() - 2 * RADIUS));
		_bar->raise();
	}

	QScrollBar* _bar = nullptr;
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


// The tables a query can read, and what each stands for. "selection" first: a
// condition written on its own reads it, so it is what a query is against
// unless it says otherwise -- and the first line of a list is where one looks
// for the ordinary case.
//
// The text forms come last: reading a cell the format could not parse is a
// deliberate act, not the everyday query.
//
// No layer is named here. The ones that exist are offered from the stack, under
// the names they carry -- a call written down here would be offered twice, and
// would name a layer the moment someone renamed it away.
// The wording of each is the one the help page uses, word for word: what the
// popup shows while typing and what the page shows when asked are the same
// sentence, so there is nothing to map from one to the other. The page may add
// a second sentence -- a line of a popup cannot carry it -- but it starts here.
static const QVector<QPair<QString, QString>> TABLES = {
    {"selection", "selection — the currently selected rows — what the listing shows"},
    {"layers", "layers — every row the layer stack lets through"},
    {"layer('')", "layer('name') — one layer, by the name it carries"},
    {"selection(text := true)", "selection(text := true) — every column as it was written"},
    {"layers(text := true)", "layers(text := true) — every column as it was written"}};

// Keywords that expect something after them, and what that something is. Typing
// one of these opens the next list on its own.
static const QHash<QString, int> FOLLOWED_BY = {
    {"FROM", 1},  {"JOIN", 1},                                              // tables
    {"SELECT", 2}, {"WHERE", 2}, {"AND", 2},    {"OR", 2},   {"BY", 2},      // columns
    {"HAVING", 2}, {"ON", 2},    {"NOT", 2},    {"COUNT", 2}, {"SUM", 2},
    {"AVG", 2},    {"MIN", 2},   {"MAX", 2},    {"DISTINCT", 2}};

// The keywords, under the heading each belongs to. Grouped rather than listed
// flat because the list is read while writing: what one looks for is a way to
// order, or to compare, and scanning an alphabet for it is what a heading
// spares. The order is the order of a query -- clauses, then what goes in them.
static const QVector<QPair<QString, QStringList>> KEYWORD_GROUPS = {
    {QT_TRANSLATE_NOOP("PVGuiQt::PVSqlCodeEditor", "Clauses"),
     {"SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "WITH",
      "AS", "DISTINCT"}},
    {QT_TRANSLATE_NOOP("PVGuiQt::PVSqlCodeEditor", "Operators"),
     {"AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "ILIKE", "IS", "NULL"}},
    {QT_TRANSLATE_NOOP("PVGuiQt::PVSqlCodeEditor", "Aggregates"),
     {"COUNT", "SUM", "AVG", "MIN", "MAX"}},
    {QT_TRANSLATE_NOOP("PVGuiQt::PVSqlCodeEditor", "Conditions"),
     {"CASE", "WHEN", "THEN", "ELSE", "END"}},
    {QT_TRANSLATE_NOOP("PVGuiQt::PVSqlCodeEditor", "Sorting"), {"ASC", "DESC"}},
    {QT_TRANSLATE_NOOP("PVGuiQt::PVSqlCodeEditor", "Sets"), {"UNION", "EXCEPT", "INTERSECT"}}};

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
	// The headings take rows of their own, so the default seven would show four
	// entries under two of them and hide the rest behind a scroll.
	_completer->setMaxVisibleItems(14);
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
	// The scroll bar is not here: completion_popup styles the one it floats,
	// on the bar itself, where a rule wins over the theme's.
	popup->setStyleSheet("QAbstractItemView { background: transparent; border: none; outline: none; }"
	                     "QAbstractItemView::item { border: none; }");
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

void PVGuiQt::PVSqlCodeEditor::set_layer_provider(std::function<QStringList()> provider)
{
	_layer_provider = std::move(provider);
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
	const int literal = layer_literal_start();
	if (literal >= 0) {
		// Inside layer('...'): what is being typed is a layer name, which the
		// prefix does not span, so both the context and what to match on are
		// taken from the literal instead.
		context = Context::Layers;
	}
	const QString typed =
	    literal >= 0 ? toPlainText().mid(literal, textCursor().position() - literal) : match;

	if (not force && literal < 0 && current_prefix().length() < 2) {
		_completer->popup()->hide();
		return;
	}

	// One category at a time, so that a heading is only written for a category
	// that has something under it. Filtered here rather than by the completer:
	// a heading matches no prefix, and would be the first thing dropped.
	struct entry {
		QString insert;
		QString label;
	};
	QVector<QPair<QString, QVector<entry>>> sections;
	const auto section = [&](const QString& title) -> QVector<entry>& {
		sections.append({title, {}});
		return sections.last().second;
	};
	const auto offer = [&](QVector<entry>& into, const QString& insert, const QString& label) {
		// On the name, which is what one types: matching the description would
		// offer entries whose beginning bears no relation to the keystrokes.
		const int dash = label.indexOf(" — ");
		const QString name = dash < 0 ? label : label.left(dash);
		if (typed.isEmpty() || name.startsWith(typed, Qt::CaseInsensitive)) {
			into.append({insert, label});
		}
	};

	if (context == Context::Layers) {
		QVector<entry>& layers = section(tr("Layers"));
		if (_layer_provider) {
			for (const QString& name : _layer_provider()) {
				offer(layers, name, name);
			}
		}
	}

	if (context == Context::Tables || context == Context::Any) {
		QVector<entry>& scopes = section(tr("Scopes"));
		for (const auto& [table, label] : TABLES) {
			offer(scopes, table, label);
		}

		// A layer is named by a string, so the whole call is offered rather
		// than the bare name: picking one writes a scope, not a fragment.
		QVector<entry>& layers = section(tr("Layers"));
		if (_layer_provider) {
			for (const QString& name : _layer_provider()) {
				const QString call = "layer('" + QString(name).replace("'", "''") + "')";
				offer(layers, call, call);
			}
		}

		QVector<entry>& others = section(tr("Sources"));
		for (const SourceCompletion& source : _sources) {
			if (source.current) {
				continue;
			}
			const QString from = source_reference(source);
			offer(others, from, from + " — " + tr("the selection of %1").arg(source.name));
			const QString all = source_reference(source, "layers");
			offer(others, all, all + " — " + tr("every row %1 lets through").arg(source.name));
		}
	}

	if (context == Context::Columns || context == Context::Any) {
		QVector<entry>& columns = section(tr("Columns"));
		offer(columns, "rowid", "rowid — " + tr("physical row index"));
		for (int i = 0; i < _column_names.size(); ++i) {
			const QString& name = _column_names.at(i);
			if (name == "rowid") {
				continue;
			}
			offer(columns, name, column_label(i));
		}

		// After the columns: they are what a position expects, and these are what
		// one wraps around them.
		QVector<entry>& conversions = section(tr("Conversions"));
		for (const auto& [call, label] : _conversions) {
			offer(conversions, call, label);
		}

		// Then the columns of the other sources. What gets inserted is the bare
		// name: in a join it is written against an alias, which only the query
		// knows -- so the source is said in the description rather than guessed
		// at in the insertion.
		for (const SourceCompletion& source : _sources) {
			if (source.current) {
				continue;
			}
			QVector<entry>& theirs = section(source.name);
			for (int i = 0; i < source.column_names.size(); ++i) {
				const QString& name = source.column_names.at(i);
				if (name == "rowid") {
					continue;
				}
				const QString type = source.column_types.value(i);
				offer(theirs, name,
				      name + " — " + (type.isEmpty() ? source.name : type + ", " + source.name));
			}
		}
	}

	if (context == Context::Any) {
		for (const auto& [title, words] : KEYWORD_GROUPS) {
			QVector<entry>& group = section(title);
			for (const QString& keyword : words) {
				offer(group, keyword, keyword);
			}
		}
	}

	_model->clear();
	int first_selectable = -1;
	// What the case that was typed points at: "sel" reaches selection and
	// SELECT alike, but only one of them is spelt that way, and that is the one
	// the keystrokes asked for.
	int first_cased = -1;
	for (const auto& [title, entries] : sections) {
		if (entries.isEmpty()) {
			continue;
		}
		auto* heading = new QStandardItem(title);
		heading->setData(title, NameRole);
		heading->setData(true, SectionRole);
		// Neither selectable nor enabled, which is also what makes the arrow
		// keys step over it.
		heading->setFlags(Qt::NoItemFlags);
		_model->appendRow(heading);

		for (const entry& item : entries) {
			auto* row = new QStandardItem(item.label);
			row->setData(item.insert, InsertRole);
			const int dash = item.label.indexOf(" — ");
			row->setData(dash < 0 ? item.label : item.label.left(dash), NameRole);
			row->setData(dash < 0 ? QString() : item.label.mid(dash + 3), DetailRole);
			_model->appendRow(row);
			const int at = _model->rowCount() - 1;
			if (first_selectable < 0) {
				first_selectable = at;
			}
			if (first_cased < 0 && not typed.isEmpty()) {
				const int dash = item.label.indexOf(" — ");
				const QString name = dash < 0 ? item.label : item.label.left(dash);
				if (name.startsWith(typed, Qt::CaseSensitive)) {
					first_cased = at;
				}
			}
		}
	}

	if (first_selectable < 0) {
		_completer->popup()->hide();
		return;
	}

	// Everything offered has already been filtered against what was typed, so
	// the completer is given an empty prefix: anything else would drop the
	// headings, which match no prefix at all.
	_completer->setCompletionPrefix(QString());
	// The entry whose own case matches goes first in line; without one, the
	// list is still offered whole and its first entry stands.
	_completer->popup()->setCurrentIndex(_completer->completionModel()->index(
	    first_cased >= 0 ? first_cased : first_selectable, 0));

	QRect rect = cursorRect();
	// sizeHintForColumn() measures the items alone: the padding and border the
	// style sheet draws around them are not in it, and a popup sized without
	// them elides the very description it exists to show.
	const QMargins frame = _completer->popup()->contentsMargins();
	// The bar floats over the list, so the room it needs is slack in the width
	// rather than a strip the layout took: without it, a description would run
	// under the bar instead of stopping short of it.
	rect.setWidth(_completer->popup()->sizeHintForColumn(0) + frame.left() + frame.right() +
	              completion_popup::BAR_WIDTH + 2 * completion_popup::BAR_INSET);
	_completer->complete(rect);
}

/**
 * Whether the cursor sits inside the string layer() takes, and where that
 * string starts.
 *
 * Read backwards from the cursor: an unmatched quote opens a literal, and the
 * word before its parenthesis says whether the literal is a layer name. Only
 * the current line is walked -- a query is written on one, and a quote left
 * open on an earlier one is a mistake rather than a context.
 */
int PVGuiQt::PVSqlCodeEditor::layer_literal_start() const
{
	const QString text = toPlainText();
	const int cursor = textCursor().position();

	int quote = -1;
	for (int i = 0; i < cursor; ++i) {
		if (text.at(i) == '\n') {
			quote = -1;
			continue;
		}
		if (text.at(i) != '\'') {
			continue;
		}
		// A doubled quote is an escaped one and closes nothing.
		if (quote >= 0 && i + 1 < cursor && text.at(i + 1) == '\'') {
			++i;
			continue;
		}
		quote = quote < 0 ? i : -1;
	}
	if (quote < 0) {
		return -1;
	}

	int pos = quote;
	while (pos > 0 && text.at(pos - 1).isSpace()) {
		--pos;
	}
	if (pos == 0 || text.at(pos - 1) != '(') {
		return -1;
	}
	--pos;
	while (pos > 0 && text.at(pos - 1).isSpace()) {
		--pos;
	}
	int start = pos;
	while (start > 0 && text.at(start - 1).isLetterOrNumber()) {
		--start;
	}
	return text.mid(start, pos - start).compare("layer", Qt::CaseInsensitive) == 0 ? quote + 1 : -1;
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
	// Inside layer('...') what is replaced is the literal, which the prefix does
	// not span: a layer name may hold spaces, and stopping at one would leave
	// half of the old name in front of the new.
	const int literal = layer_literal_start();
	cursor.setPosition(literal >= 0 ? literal : cursor.position() - current_prefix().length(),
	                   QTextCursor::KeepAnchor);
	cursor.insertText(inserted);
	if (literal >= 0) {
		// Close what was opened, unless the query already carries the closing
		// quote -- typing inside a complete call is how a name gets corrected.
		const QString rest = toPlainText().mid(cursor.position());
		if (not rest.startsWith('\'')) {
			cursor.insertText("')");
		}
	}

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
	// layer('') is offered as a shape to fill rather than a layer: naming one
	// of them there would be picking for the user, and picking wrong as soon as
	// the view holds more than the base layer. The caret goes between the
	// quotes and the layers are offered straight away.
	const bool wants_layer = inserted.endsWith("('')");
	if (wants_layer) {
		cursor.setPosition(cursor.position() - 2);
	}
	setTextCursor(cursor);
	_inserting = false;

	if (opens_next) {
		show_completions(true);
	}
	if (wants_layer) {
		// Deferred by one turn of the loop: the popup is being hidden as part of
		// accepting this completion, and a list opened before that lands would
		// be shut again the moment it appeared.
		QTimer::singleShot(0, this, [this]() { show_completions(true); });
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
