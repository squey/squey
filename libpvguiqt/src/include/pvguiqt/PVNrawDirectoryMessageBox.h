/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2017
 */

#ifndef __PVGUIQT_PVNRAWDIRECTORYMESSAGEBOX_H__
#define __PVGUIQT_PVNRAWDIRECTORYMESSAGEBOX_H__

#include <QMessageBox>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/rush/PVNraw.h>

class PVNrawDirectoryMessageBox : public QMessageBox
{
  public:
	PVNrawDirectoryMessageBox(QWidget* parent = nullptr) : QMessageBox(parent)
	{
		const QString& config_nraw_tmp = QString::fromStdString(PVRush::PVNraw::config_nraw_tmp);
		QString nraw_tmp = PVCore::PVConfig::get().config().value(config_nraw_tmp).toString();

		if (QFileInfo(nraw_tmp).isWritable()) {
			return;
		}

		setStandardButtons(QMessageBox::Ok);
		setIcon(QMessageBox::Information);
		setWindowTitle("Temporary working directory error");
		setText(
		    QString(
		        "The temporary working directory <b>%1</b> does not exists or is not writable. "
		        "Please, change the permissions or chose another directory.<br><br>"
				"This directory emplacement needs to be specified once and for all."
		        "A fast underlying storage for this directory will provide better performances.")
		        .arg(nraw_tmp));

		QAbstractButton* browser_button = addButton(tr("Browse..."), QMessageBox::ActionRole);

		do {
			exec();

			if (clickedButton() == browser_button) {
				nraw_tmp = PVWidgets::PVFileDialog::getExistingDirectory();
				if (not nraw_tmp.isEmpty()) {
					PVCore::PVConfig::get().config().setValue(config_nraw_tmp, nraw_tmp);
				}
			}

		} while (not QFileInfo(nraw_tmp).isWritable());
	}
};

#endif // __PVGUIQT_PVNRAWDIRECTORYMESSAGEBOX_H__
