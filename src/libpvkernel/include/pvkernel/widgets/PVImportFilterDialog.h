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

#ifndef __PVWIDGETS_PVIMPORTFILTERDIALOG_H__
#define __PVWIDGETS_PVIMPORTFILTERDIALOG_H__

#include <QDialog>
#include <QString>
#include <QStringList>

#include <functional>
#include <optional>
#include <vector>

class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QTreeWidget;

namespace PVWidgets
{

/**
 * The columns and the rows to import from files: the user checks the columns among those found,
 * and may give a SQL condition that the rows to import hold.
 */
class PVImportFilterDialog : public QDialog
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
	/**
	 * Starts with the columns named @p checked checked, or all of them if none, and with the
	 * condition @p row_filter. The dialog accepts no condition that @p error_of finds wrong.
	 */
	PVImportFilterDialog(const std::vector<Column>& columns,
	                     const QStringList& checked,
	                     const QString& row_filter,
	                     error_of_t error_of,
	                     QWidget* parent = nullptr);

  public:
	/** The names of the columns checked, or none if all of them are. */
	QStringList columns() const;

	/** The condition that the rows to import hold, or none for all of them. */
	QString row_filter() const;

  public Q_SLOTS:
	void accept() override;

  private:
	void update_count();

  private:
	error_of_t _error_of;
	QLineEdit* _search;
	QTreeWidget* _columns;
	QLabel* _count;
	QPlainTextEdit* _row_filter;
	QLabel* _error;
	QPushButton* _ok;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVIMPORTFILTERDIALOG_H__
