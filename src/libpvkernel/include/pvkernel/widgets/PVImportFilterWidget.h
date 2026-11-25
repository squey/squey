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

#ifndef __PVWIDGETS_PVIMPORTFILTERWIDGET_H__
#define __PVWIDGETS_PVIMPORTFILTERWIDGET_H__

#include <QString>
#include <QStringList>
#include <QTimer>
#include <QWidget>

#include <functional>
#include <optional>
#include <vector>

class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QTreeWidget;

namespace PVWidgets
{

/**
 * The columns and the rows to import from files: the user checks the columns among those found,
 * and may give a SQL condition that the rows to import hold, which is checked as it is written.
 */
class PVImportFilterWidget : public QWidget
{
	Q_OBJECT

  public:
	struct Column {
		QString name;
		QString type;
		/** The share of the rows that give it a value, from 0 to 1, if it is known. */
		std::optional<double> fill_rate;
	};

	/** What is wrong with a condition on the rows, or nothing. */
	using error_of_t = std::function<QString(const QString& condition)>;

  public:
	explicit PVImportFilterWidget(QWidget* parent = nullptr);

  public:
	/**
	 * Lists @p columns, those named @p checked checked, or all of them if none, with the
	 * condition @p row_filter, which @p error_of checks.
	 */
	void set_columns(const std::vector<Column>& columns,
	                 const QStringList& checked,
	                 const QString& row_filter,
	                 error_of_t error_of);

	/** The names of the columns checked, or none if all of them are. */
	QStringList columns() const;

	/** The condition that the rows to import hold, or none for all of them. */
	QString row_filter() const;

	/** Whether a column is checked, and the condition is checked and found right. */
	bool is_valid() const;

	/** Checks the condition at once, rather than once the user stops writing it. */
	void check_row_filter();

  Q_SIGNALS:
	/** The columns checked, the condition or what is wrong with it changed. */
	void changed();

  private:
	void update_count();

  private:
	error_of_t _error_of;
	QLineEdit* _search;
	QTreeWidget* _columns;
	QLabel* _count;
	QPlainTextEdit* _row_filter;
	QLabel* _error;
	QTimer _check_timer;
	bool _row_filter_checked = false;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVIMPORTFILTERWIDGET_H__
