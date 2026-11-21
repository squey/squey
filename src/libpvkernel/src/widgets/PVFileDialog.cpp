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

#include <pvkernel/core/PVWSLHelper.h>

#include <QAbstractItemModel>
#include <QDialog>
#include <QDir>
#include <QString>

#include <string>
#include <utility>
#include <vector>

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
                                                 Options options /* = Options() */,
                                                 const QString& default_suffix /* = QString() */
                                                 )
{
	const QStringList schemes = QStringList(QStringLiteral("file"));
	const QUrl selectedUrl = getSaveFileUrl(parent, caption, QUrl::fromLocalFile(dir), filter,
	                                        selectedFilter, options, schemes, default_suffix);
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
	dialog.setOptions(options);
	dialog.setSupportedSchemes(supportedSchemes);
	if (selectedFilter && !selectedFilter->isEmpty())
		dialog.selectNameFilter(*selectedFilter);
	if (exec(dialog) == QDialog::Accepted) {
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
	dialog.setOptions(options);
	dialog.setFileMode(QFileDialog::ExistingFiles);
	dialog.setSupportedSchemes(supportedSchemes);
	if (selectedFilter && !selectedFilter->isEmpty())
		dialog.selectNameFilter(*selectedFilter);
	if (exec(dialog) == QDialog::Accepted) {
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
	dialog.setOptions(options);
	dialog.setFileMode(QFileDialog::Directory);
	dialog.setSupportedSchemes(supportedSchemes);
	if (exec(dialog) == QDialog::Accepted)
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
    const QStringList& supportedSchemes /* = QStringList() */,
    const QString& default_suffix /* = QString() */
    )
{
	QFileDialog dialog(parent, caption, dir.toLocalFile(), filter);
	dialog.setOptions(options);
	dialog.setFileMode(QFileDialog::AnyFile);
	dialog.setSupportedSchemes(supportedSchemes);
	dialog.setAcceptMode(QFileDialog::AcceptSave);
	dialog.setDefaultSuffix(default_suffix);
	if (selectedFilter && !selectedFilter->isEmpty())
		dialog.selectNameFilter(*selectedFilter);
	if (exec(dialog) == QDialog::Accepted) {
		if (selectedFilter)
			*selectedFilter = dialog.selectedNameFilter();
		return dialog.selectedUrls().value(0);
	}
	return {};
}

int PVWidgets::PVFileDialog::exec(QFileDialog& dialog)
{
	customize_for_wsl(dialog);
	const int result = dialog.exec();

	// Children are destroyed in the order they were added, and the dialog of Qt adds its file
	// system model before the views showing it. A model resets the persistent indexes it tracks as
	// it is destroyed, but not those it lost track of: on Windows, QFileSystemModel renames a
	// directory to its case on disk without updating the visible children of its parent, and
	// sort() then moves the persistent indexes of that directory to row -1, out of its table. A
	// view releasing one of them after the model reads freed memory.
	for (QAbstractItemModel* model :
	     dialog.findChildren<QAbstractItemModel*>(Qt::FindDirectChildrenOnly)) {
		// Taken off and back on, the model goes to the end of the list.
		model->setParent(nullptr);
		model->setParent(&dialog);
	}

	return result;
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
