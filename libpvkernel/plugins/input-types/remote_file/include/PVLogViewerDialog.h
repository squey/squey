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

#ifndef LOGVIEWERDIALOG_FILE_H
#define LOGVIEWERDIALOG_FILE_H

#include "logviewerwidget.h"
#include <QDialog>
#include <QHash>
#include <QUrl>
#include <QComboBox>

class PVLogViewerDialog : public QDialog
{
	Q_OBJECT
  public:
	PVLogViewerDialog(QStringList const& formats, QWidget* parent);
	virtual ~PVLogViewerDialog();

  public:
	QHash<QString, QUrl> const& getDlFiles() { return _dl_files; }
	QString getSelFormat();

  public Q_SLOTS:
	void slotDownloadFiles();

  protected:
	LogViewerWidget* pv_RemoteLog;
	QHash<QString, QUrl> _dl_files;
	QComboBox* _combo_format;
	QString _format;
};

#endif
