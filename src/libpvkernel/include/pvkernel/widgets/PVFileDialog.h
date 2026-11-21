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
#include <QList>
#include <QString>
#include <QStringList>
#include <QUrl>

class QWidget;

namespace PVWidgets
{

/**
 * The file dialogs of Squey: those of the platform where it has some, the one of Qt elsewhere.
 *
 * The functions are the static ones of QFileDialog. The dialogs of the platforms hold nothing
 * but files: what else a dialog asks for is asked before or after it.
 */
class PVFileDialog
{
  public:
	using Options = QFileDialog::Options;

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
	                                    Options options = QFileDialog::ShowDirsOnly);

	/** @param default_suffix What a name typed without a suffix takes, as QFileDialog has it. */
	static QString getSaveFileName(QWidget* parent = nullptr,
	                               const QString& caption = QString(),
	                               const QString& dir = QString(),
	                               const QString& filter = QString(),
	                               QString* selectedFilter = nullptr,
	                               Options options = Options(),
	                               const QString& default_suffix = QString());

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
	                           const QStringList& supportedSchemes = QStringList(),
	                           const QString& default_suffix = QString());

  private:
	static int exec(QFileDialog& dialog);
	static void wait_for_portal();
	static void customize_for_wsl(QFileDialog& dialog);
};

} // namespace PVWidgets

#endif //__PVGUIQT_PVFILEDIALOG_H__
