/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __PVWIDGETS_PVMULTIPLEFILEDIALOG_H__
#define __PVWIDGETS_PVMULTIPLEFILEDIALOG_H__

#include <pvkernel/widgets/PVFileDialog.h>

#include <QGroupBox>
#include <QListWidget>
#include <QLineEdit>
#include <QGridLayout>
#include <QPushButton>

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