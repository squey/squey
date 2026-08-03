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

#include <QHash>
#include <QPair>
#include <QString>
#include <QStringList>
#include <QTextEdit>
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
class PVSqlCodeEditor : public QTextEdit
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
	enum class Context { Any, Tables, Columns };

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
	//! Bare name -> quoted form, for the names a query cannot carry as-is.
	QHash<QString, QString> _quoted_form;
	//! Set while inserting a completion, to keep that edit from re-triggering.
	bool _inserting = false;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVSQLCODEEDITOR_H__
