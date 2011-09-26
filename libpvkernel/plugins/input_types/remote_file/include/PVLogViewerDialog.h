#ifndef LOGVIEWERDIALOG_FILE_H
#define LOGVIEWERDIALOG_FILE_H

#include "logviewerwidget.h"
#include <QDialog>
#include <QHash>
#include <QUrl>

class PVLogViewerDialog: public QDialog
{
	Q_OBJECT
public:
	PVLogViewerDialog(QWidget* parent);
	virtual ~PVLogViewerDialog();

public:
	QHash<QString, QUrl> const& getDlFiles() { return _dl_files; }

public slots:
	void slotDownloadFiles();

protected:
	LogViewerWidget* pv_RemoteLog;
	QHash<QString, QUrl> _dl_files;
};

#endif
