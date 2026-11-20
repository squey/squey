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

#ifndef __PVWIDGETS_PVIMPORTFILESWIDGET_H__
#define __PVWIDGETS_PVIMPORTFILESWIDGET_H__

#include <QString>
#include <QStringList>
#include <QWidget>

#include <vector>

class QListWidget;
class QPushButton;
class QRadioButton;

namespace PVWidgets
{

/**
 * The files of an import: their list, buttons to add and to remove some, and whether several
 * files make one source or one each.
 */
class PVImportFilesWidget : public QWidget
{
	Q_OBJECT

  public:
	/** With @p dialog_title and @p dialog_filter for the dialog that adds files. */
	PVImportFilesWidget(QString dialog_title, QString dialog_filter, QWidget* parent = nullptr);

  public:
	QStringList const& paths() const { return _paths; }
	void set_paths(QStringList const& paths);
	/** Adds those of @p paths that are not in the list yet, after the others. */
	void add_paths(QStringList const& paths);
	bool one_source_per_file() const;
	/** The paths of the files of each source. */
	std::vector<QStringList> sources() const;

  Q_SIGNALS:
	void paths_changed();
	/** Whether the files make one source or one each changed. */
	void sources_changed();

  private:
	void add_files();
	void remove_files();

  private:
	QString _dialog_title;
	QString _dialog_filter;
	QStringList _paths;
	QListWidget* _files;
	QPushButton* _remove;
	QWidget* _sources;
	QRadioButton* _source_per_file;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVIMPORTFILESWIDGET_H__
