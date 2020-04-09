/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVGUIQT_PVFILEDIALOG_H__
#define __PVGUIQT_PVFILEDIALOG_H__

#include <QFileDialog>

#include <pvkernel/core/PVWSLHelper.h>

namespace PVWidgets
{

class PVFileDialog : public QFileDialog
{
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
