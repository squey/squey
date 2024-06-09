/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef __PVWIDGETS_PVMULTIPLEFILEDIALOG_H__
#define __PVWIDGETS_PVMULTIPLEFILEDIALOG_H__

#include <pvkernel/widgets/PVFileDialog.h>
#include <qcontainerfwd.h>
#include <qlist.h>
#include <qstring.h>
#include <qurl.h>
#include <QGroupBox>
#include <QListWidget>
#include <QLineEdit>
#include <QGridLayout>
#include <QPushButton>

class QLineEdit;
class QListWidget;
class QPushButton;
class QWidget;

namespace PVWidgets
{

class PVMultipleFileDialog : public PVFileDialog
{
  public:
	PVMultipleFileDialog(QWidget* parent = nullptr)
	    : PVFileDialog(parent)
	{
	}

	PVMultipleFileDialog(QWidget* parent = nullptr,
	                     const QString& caption = QString(),
	                     const QString& directory = QString(),
	                     const QString& filter = QString());

	QList<QUrl> selectedUrls() const;

	static QStringList getOpenFileNames(QWidget* parent = nullptr,
	                                    const QString& caption = QString(),
	                                    const QString& dir = QString(),
	                                    const QString& filter = QString(),
	                                    QString* selectedFilter = nullptr,
	                                    Options options = Options());

	static QList<QUrl> getOpenFileUrls(QWidget* parent = nullptr,
	                                   const QString& caption = QString(),
	                                   const QUrl& dir = QString(),
	                                   const QString& filter_string = QString(),
	                                   QString* selectedFilter = nullptr,
	                                   Options options = Options(),
	                                   const QStringList& supportedSchemes = QStringList());

  private:
	QPushButton* _add_button;
	QPushButton* _remove_button;
	QPushButton* _open_button;
	QListWidget* _files_list;
	QLineEdit* _filename_edit;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVMULTIPLEFILEDIALOG_H__
