//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVFileDialog.h>
#include <qdialog.h>
#include <qdir.h>
#include <qstringliteral.h>
#include <string>
#include <utility>
#include <vector>

#include "pvkernel/core/PVWSLHelper.h"

class QWidget;

PVWidgets::PVFileDialog::PVFileDialog(QWidget* parent, Qt::WindowFlags flags)
    : QFileDialog(parent, flags)
{
	QFileDialog::setOption(DontUseNativeDialog, true);
	customize_for_wsl(*this);
}

PVWidgets::PVFileDialog::PVFileDialog(QWidget* parent /* = nullptr */,
                                      const QString& caption /* = QString() */,
                                      const QString& directory /* = QString() */,
                                      const QString& filter /* = QString()*/)
    : QFileDialog(parent, caption, directory, filter)
{
	QFileDialog::setOption(DontUseNativeDialog, true);
	customize_for_wsl(*this);
}

QString PVWidgets::PVFileDialog::getOpenFileName(QWidget* parent /* = nullptr */,
                                                 const QString& caption /* = QString() */,
                                                 const QString& dir /* = QString() */,
                                                 const QString& filter /* = QString() */,
                                                 QString* selectedFilter /* = nullptr */,
                                                 Options options /* = Options() */
                                                 )
{
	const QStringList schemes = QStringList(QStringLiteral("file"));
	const QUrl selectedUrl = getOpenFileUrl(parent, caption, QUrl::fromLocalFile(dir), filter,
	                                        selectedFilter, options, schemes);
	return selectedUrl.toLocalFile();
}

QStringList PVWidgets::PVFileDialog::getOpenFileNames(QWidget* parent /* = nullptr */,
                                                      const QString& caption /* = QString() */,
                                                      const QString& dir /* = QString() */,
                                                      const QString& filter /* = QString() */,
                                                      QString* selectedFilter /* = nullptr */,
                                                      Options options /* = Options() */
                                                      )
{
	const QStringList schemes = QStringList(QStringLiteral("file"));
	const QList<QUrl> selectedUrls = getOpenFileUrls(parent, caption, QUrl::fromLocalFile(dir),
	                                                 filter, selectedFilter, options, schemes);
	QStringList fileNames;
	fileNames.reserve(selectedUrls.size());
	for (const QUrl& url : selectedUrls)
		fileNames << url.toLocalFile();
	return fileNames;
}

QString PVWidgets::PVFileDialog::getExistingDirectory(QWidget* parent /* = nullptr */,
                                                      const QString& caption /* = QString() */,
                                                      const QString& dir /* = QString() */,
                                                      Options options /* = ShowDirsOnly */
                                                      )
{
	const QStringList schemes = QStringList(QStringLiteral("file"));
	const QUrl selectedUrl =
	    getExistingDirectoryUrl(parent, caption, QUrl::fromLocalFile(dir), options, schemes);
	return selectedUrl.toLocalFile();
}

QString PVWidgets::PVFileDialog::getSaveFileName(QWidget* parent /* = nullptr */,
                                                 const QString& caption /* = QString() */,
                                                 const QString& dir /* = QString() */,
                                                 const QString& filter /* = QString() */,
                                                 QString* selectedFilter /* = nullptr */,
                                                 Options options /* = Options() */
                                                 )
{
	const QStringList schemes = QStringList(QStringLiteral("file"));
	const QUrl selectedUrl = getSaveFileUrl(parent, caption, QUrl::fromLocalFile(dir), filter,
	                                        selectedFilter, options, schemes);
	return selectedUrl.toLocalFile();
}

QUrl PVWidgets::PVFileDialog::getOpenFileUrl(
    QWidget* parent /* = nullptr */,
    const QString& caption /* = QString() */,
    const QUrl& dir /* = QString() */,
    const QString& filter_string /* = QString() */,
    QString* selectedFilter /* = nullptr */,
    Options options /* = Options() */,
    const QStringList& supportedSchemes /* = QStringList() */)
{
	QFileDialog dialog(parent, caption, dir.toLocalFile(), filter_string);
	dialog.setOptions(get_options(options));
	customize_for_wsl(dialog);
	dialog.setSupportedSchemes(supportedSchemes);
	if (selectedFilter && !selectedFilter->isEmpty())
		dialog.selectNameFilter(*selectedFilter);
	if (dialog.exec() == QDialog::Accepted) {
		if (selectedFilter)
			*selectedFilter = dialog.selectedNameFilter();
		return dialog.selectedUrls().value(0);
	}
	return {};
}

QList<QUrl>
PVWidgets::PVFileDialog::getOpenFileUrls(QWidget* parent /* = nullptr */,
                                         const QString& caption /* = QString() */,
                                         const QUrl& dir /* = QString() */,
                                         const QString& filter_string /* = QString() */,
                                         QString* selectedFilter /* = nullptr */,
                                         Options options /* = Options() */,
                                         const QStringList& supportedSchemes /* = QStringList() */)
{
	QFileDialog dialog(parent, caption, dir.toLocalFile(), filter_string);
	dialog.setOptions(get_options(options));
	dialog.setFileMode(QFileDialog::ExistingFiles);
	customize_for_wsl(dialog);
	dialog.setSupportedSchemes(supportedSchemes);
	if (selectedFilter && !selectedFilter->isEmpty())
		dialog.selectNameFilter(*selectedFilter);
	if (dialog.exec() == QDialog::Accepted) {
		if (selectedFilter)
			*selectedFilter = dialog.selectedNameFilter();
		return dialog.selectedUrls();
	}
	return {};
}

QUrl PVWidgets::PVFileDialog::getExistingDirectoryUrl(
    QWidget* parent /* = nullptr */,
    const QString& caption /* = QString() */,
    const QUrl& dir /* = QString() */,
    Options options /* = Options() */,
    const QStringList& supportedSchemes /* = QStringList() */)
{
	QFileDialog dialog(parent, caption, dir.toLocalFile());
	dialog.setOptions(get_options(options));
	dialog.setFileMode(QFileDialog::Directory);
	customize_for_wsl(dialog);

	dialog.setSupportedSchemes(supportedSchemes);
	if (dialog.exec() == QDialog::Accepted)
		return dialog.selectedUrls().value(0);
	return {};
}

QUrl PVWidgets::PVFileDialog::getSaveFileUrl(
    QWidget* parent /* = nullptr */,
    const QString& caption /* = QString() */,
    const QUrl& dir /* = QUrl() */,
    const QString& filter /* = QString() */,
    QString* selectedFilter /* = nullptr */,
    Options options /* = Options() */,
    const QStringList& supportedSchemes /* = QStringList() */
    )
{
	QFileDialog dialog(parent, caption, dir.toLocalFile(), filter);
	dialog.setOptions(get_options(options));
	dialog.setFileMode(QFileDialog::AnyFile);
	customize_for_wsl(dialog);
	dialog.setSupportedSchemes(supportedSchemes);
	dialog.setAcceptMode(AcceptSave);
	if (selectedFilter && !selectedFilter->isEmpty())
		dialog.selectNameFilter(*selectedFilter);
	if (dialog.exec() == QDialog::Accepted) {
		if (selectedFilter)
			*selectedFilter = dialog.selectedNameFilter();
		return dialog.selectedUrls().value(0);
	}
	return {};
}

void PVWidgets::PVFileDialog::setOptions(Options options)
{
	QFileDialog::setOptions(get_options(options));
}

void PVWidgets::PVFileDialog::customize_for_wsl(QFileDialog& dialog)
{
	// https://stackoverflow.com/questions/44568065/how-to-add-custom-items-to-qfiledialog

	if (PVCore::PVWSLHelper::is_microsoft_wsl()) {

		if (dialog.directory().isEmpty()) {
			dialog.setDirectory(QString::fromStdString(PVCore::PVWSLHelper::user_directory()));
		}

		QList<QUrl> drives;

		drives << QUrl::fromLocalFile(
		    QDir(QString::fromStdString(PVCore::PVWSLHelper::user_directory())).absolutePath());
		drives << QUrl::fromLocalFile(QDir("/").absolutePath());
		for (const std::pair<std::string, std::string>& drive :
		     PVCore::PVWSLHelper::drives_list()) {
			drives << QUrl::fromLocalFile(
			    QDir(QString::fromStdString(drive.second)).absolutePath());
		}

		dialog.setSidebarUrls(drives);
	}
}

QFileDialog::Options PVWidgets::PVFileDialog::get_options(const Options& options)
{
	return options | DontUseNativeDialog;
}
