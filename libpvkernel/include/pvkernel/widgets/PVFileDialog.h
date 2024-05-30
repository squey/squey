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

#ifndef __PVGUIQT_PVFILEDIALOG_H__
#define __PVGUIQT_PVFILEDIALOG_H__

#include <QFileDialog>

#include <pvkernel/core/PVWSLHelper.h>

namespace PVWidgets
{

class PVFileDialog : public QFileDialog
{
	Q_OBJECT

  public:
	PVFileDialog(QWidget* parent, Qt::WindowFlags flags);

	PVFileDialog(QWidget* parent = nullptr,
	             const QString& caption = QString(),
	             const QString& directory = QString(),
	             const QString& filter = QString());

	virtual ~PVFileDialog(){};

  public:
	static QString getOpenFileName(QWidget* parent = nullptr,
	                               const QString& caption = QString(),
	                               const QString& dir = QString(),
	                               const QString& filter = QString(),
	                               QString* selectedFilter = nullptr,
	                               Options options = Options());

	static QStringList getOpenFileNames(QWidget* parent = nullptr,
	                                    const QString& caption = QString(),
	                                    const QString& dir = QString(),
	                                    const QString& filter = QString(),
	                                    QString* selectedFilter = nullptr,
	                                    Options options = Options());

	static QString getExistingDirectory(QWidget* parent = nullptr,
	                                    const QString& caption = QString(),
	                                    const QString& dir = QString(),
	                                    Options options = ShowDirsOnly);

	static QString getSaveFileName(QWidget* parent = nullptr,
	                               const QString& caption = QString(),
	                               const QString& dir = QString(),
	                               const QString& filter = QString(),
	                               QString* selectedFilter = nullptr,
	                               Options options = Options());

	static QUrl getOpenFileUrl(QWidget* parent = nullptr,
	                           const QString& caption = QString(),
	                           const QUrl& dir = QString(),
	                           const QString& filter_string = QString(),
	                           QString* selectedFilter = nullptr,
	                           Options options = Options(),
	                           const QStringList& supportedSchemes = QStringList());

	static QList<QUrl> getOpenFileUrls(QWidget* parent = nullptr,
	                                   const QString& caption = QString(),
	                                   const QUrl& dir = QString(),
	                                   const QString& filter_string = QString(),
	                                   QString* selectedFilter = nullptr,
	                                   Options options = Options(),
	                                   const QStringList& supportedSchemes = QStringList());

	static QUrl getExistingDirectoryUrl(QWidget* parent = nullptr,
	                                    const QString& caption = QString(),
	                                    const QUrl& dir = QString(),
	                                    Options options = Options(),
	                                    const QStringList& supportedSchemes = QStringList());

	static QUrl getSaveFileUrl(QWidget* parent = nullptr,
	                           const QString& caption = QString(),
	                           const QUrl& dir = QUrl(),
	                           const QString& filter = QString(),
	                           QString* selectedFilter = nullptr,
	                           Options options = Options(),
	                           const QStringList& supportedSchemes = QStringList());

	void setOptions(Options options);

  protected:
	static void customize_for_wsl(QFileDialog& dialog);
	static Options get_options(const Options& options);
};

} // namespace PVWidgets

#endif //__PVGUIQT_PVFILEDIALOG_H__
