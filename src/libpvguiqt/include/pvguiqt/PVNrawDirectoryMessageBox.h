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
				"This directory emplacement needs to be specified once and for all.<br>"
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
