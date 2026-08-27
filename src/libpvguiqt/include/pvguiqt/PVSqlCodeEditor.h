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

#ifndef __PVGUIQT_PVSQLCODEEDITOR_H__
#define __PVGUIQT_PVSQLCODEEDITOR_H__

#include <pvguiqt/export.h>

#include <QHash>
#include <QPair>
#include <QString>
#include <QStringList>
#include <QTextEdit>

#include <functional>
#include <QVector>

class QCompleter;
class QStandardItemModel;

namespace PVGuiQt
{

/**
 * SQL editor with syntax highlighting and context-aware completion.
 *
 * Completion is driven by what precedes the cursor rather than offering
 * everything at once: after FROM only the tables make sense, after WHERE only
 * columns do. The popup also opens by itself once a keyword that expects a
 * follow-up has been typed, and again when a column name is clicked, so writing
 * a query is a sequence of choices rather than something to be recalled --
 * which is what the plain everything-in-one-list completion failed to give.
 */
class PVGUIQT_EXPORT PVSqlCodeEditor : public QTextEdit
{
	Q_OBJECT

  public:
	/**
	 * Role holding the text a completion inserts.
	 *
	 * It cannot be the edit role: QStandardItem stores that one and the display
	 * role in the same place, so writing the insertion there would overwrite the
	 * description the popup is meant to show -- and a bare list of names is
	 * exactly what the descriptions exist to improve on.
	 */
	static constexpr int InsertRole = Qt::UserRole + 1;

	//! The name alone, which the popup shows first and in full strength.
	static constexpr int NameRole = Qt::UserRole + 2;
	//! What the name holds -- a type, a source -- shown beside it, dimmed.
	static constexpr int DetailRole = Qt::UserRole + 3;
	//! Set on the heading of a category rather than on something to insert.
	static constexpr int SectionRole = Qt::UserRole + 4;

	explicit PVSqlCodeEditor(QWidget* parent = nullptr);

  public:
	/**
	 * Replace the columns offered by the completer.
	 *
	 * @param names axis names, verbatim
	 * @param types SQL type of each, shown beside the name; may be empty
	 * @param axis_types Squey's own type for each -- "ipv4", "datetime" -- which
	 *                   is what says an address is an address rather than a
	 *                   number, and which conversions are worth offering. May be
	 *                   empty.
	 */
	void set_columns(const QStringList& names,
	                 const QStringList& types = {},
	                 const QStringList& axis_types = {});

	/**
	 * A source a query can name, as the completer needs to offer it.
	 *
	 * Held apart from set_columns(): those are the columns of the console's own
	 * source, the ones a bare name reaches, whereas these have to be written
	 * with the source they come from.
	 */
	struct SourceCompletion {
		QString name;
		//! Which one among its namesakes, for source_position :=.
		int position = 0;
		//! True while "name.selection" resolves; a namesake has no schema.
		bool has_schema = false;
		//! True for the source the console sits on, whose columns are bare.
		bool current = false;
		QStringList column_names;
		QStringList column_types;
	};

	/**
	 * Replace the sources offered by the completer.
	 *
	 * What this adds over set_columns() is everything a query cannot guess: the
	 * names of the other sources, which of them the short schema form reaches,
	 * and the columns each holds -- a listing shows the names, nothing shows
	 * the columns.
	 */
	void set_sources(const QVector<SourceCompletion>& sources);

	/**
	 * Replace the functions the completer offers, as name and description.
	 *
	 * There are hundreds, so they are only offered once enough has been typed
	 * to narrow them: with nothing typed they would bury the columns and the
	 * scopes, which are what one reaches for far more often. That is also why
	 * they come last of what fits where a value goes.
	 */
	void set_functions(const std::vector<std::pair<std::string, std::string>>& functions);

	/**
	 * Where the completer reads the layers layer('name') can be given.
	 *
	 * A function rather than a list: layers are created, renamed and dropped
	 * while the console stays open, so anything captured once would go on
	 * offering names that no longer resolve.
	 */
	void set_layer_provider(std::function<QStringList()> provider);

	/**
	 * Keep @a query as the last one run, for Up to reach.
	 *
	 * Whatever was run, not whatever worked: a query one wants back is most
	 * often one that failed, and going and fixing it is the point. A run
	 * repeating the one before it is not kept twice.
	 *
	 * The list lives with the console and goes with it -- nothing is written to
	 * disk, and nothing of it is saved into the investigation.
	 */
	void remember(const QString& query);

	//! What Up walks back through, oldest first. For a test to read.
	const QStringList& history() const { return _history; }

	/**
	 * One line tall.
	 *
	 * A query is usually one line, and the console is a strip at the bottom of
	 * the window: asking for more would take room from what the query is written
	 * about. The dock can be pulled up for a query that needs it, which is why
	 * this is a hint rather than a fixed height.
	 */
	QSize sizeHint() const override;
	QSize minimumSizeHint() const override;

  Q_SIGNALS:
	//! Enter was pressed on a query. Ctrl+Enter and Shift+Enter break the line.
	void run_requested();

  protected:
	void keyPressEvent(QKeyEvent* event) override;
	void focusInEvent(QFocusEvent* event) override;
	//! Clicking a column name offers the other columns; see complete_column_at_cursor().
	void mouseReleaseEvent(QMouseEvent* event) override;
	//! Paste as plain text: rich text would carry formatting into the query.
	void insertFromMimeData(const QMimeData* source) override;

  private:
	//! What makes sense at the cursor, derived from the preceding keyword.
	enum class Context { Any, Tables, Columns, Layers };

	//! Paint the popup the way the rest of the console looks, in either theme.
	void restyle_popup();

	/**
	 * Where the layer name being typed starts, or -1 outside one.
	 *
	 * A layer name is a string literal, so it holds what an identifier cannot
	 * -- spaces above all -- and neither the word under the cursor nor the
	 * completion prefix spans it. Both what to offer and what an accepted
	 * completion replaces are measured from here.
	 */
	int layer_literal_start() const;

	Context context_at_cursor() const;
	//! What the popup shows for a column: its name, and what it really holds.
	QString column_label(int index) const;
	//! Word being typed under the cursor, which is what gets completed.
	QString current_prefix() const;
	void insert_completion(const QModelIndex& completion);
	//! Repopulate for the current context and pop up. @a force ignores the
	//! minimum prefix length, which is how an empty position still offers a list.
	void show_completions(bool force);
	/**
	 * @param context which names to offer
	 * @param match what the offered names must start with; empty offers them all,
	 *              whatever the cursor sits on
	 */
	void show_completions(bool force, Context context, const QString& match);
	/**
	 * Offer the columns when the cursor sits on the name of one.
	 *
	 * Returns false when it does not, leaving the click to mean what a click
	 * ordinarily means.
	 */
	bool complete_column_at_cursor();

  private:
	QCompleter* _completer = nullptr;
	QStandardItemModel* _model = nullptr;
	QStringList _column_names;
	QStringList _column_types;
	QStringList _column_axis_types;
	//! The conversions worth offering, derived from the axis types present.
	QVector<QPair<QString, QString>> _conversions;
	//! The other sources a query can name. Empty while there is only one.
	QVector<SourceCompletion> _sources;
	//! Insert form and label of every function offered. See set_functions().
	QVector<QPair<QString, QString>> _functions;
	//! How much of a name has to be typed before the functions are offered.
	static constexpr int FUNCTION_PREFIX = 2;
	//! Asked for the layers whenever a list is built. See set_layer_provider().
	std::function<QStringList()> _layer_provider;
	//! Bare name -> quoted form, for the names a query cannot carry as-is.
	QHash<QString, QString> _quoted_form;
	//! Set while inserting a completion, to keep that edit from re-triggering.
	bool _inserting = false;

	/**
	 * Walk to the query @a delta steps away and put it in the editor.
	 *
	 * @return false when there is nowhere to go, so the key can fall through to
	 *         what it otherwise does.
	 */
	bool recall(int delta);

	//! The queries run in this console, oldest first. See remember().
	QStringList _history;
	/**
	 * Which entry the editor is standing on, or -1 for the draft.
	 *
	 * The draft is a position of its own, just past the newest entry: Up steps
	 * off it and Down comes back to it. Typing puts one back on it, which is
	 * what makes Up save the text before replacing it.
	 */
	int _history_at = -1;
	//! What was being typed when Up was first pressed, kept for Down.
	QString _draft;
	//! Set while recall() writes, so its own edit is not read as typing.
	bool _recalling = false;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVSQLCODEEDITOR_H__
