/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#include <pvkernel/widgets/PVFileDialog.h>

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
	return QUrl();
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
	return QList<QUrl>();
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
	dialog.setFileMode(options & ShowDirsOnly ? DirectoryOnly : Directory);

	dialog.setSupportedSchemes(supportedSchemes);
	if (dialog.exec() == QDialog::Accepted)
		return dialog.selectedUrls().value(0);
	return QUrl();
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
	return QUrl();
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
