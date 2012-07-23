/**
 * \file PVLogViewerDialog.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef LOGVIEWERDIALOG_FILE_H
#define LOGVIEWERDIALOG_FILE_H

#include "logviewerwidget.h"
#include <QDialog>
#include <QHash>
#include <QUrl>
#include <QComboBox>

class PVLogViewerDialog: public QDialog
{
	Q_OBJECT
public:
	PVLogViewerDialog(QStringList const& formats, QWidget* parent);
	virtual ~PVLogViewerDialog();

public:
	QHash<QString, QUrl> const& getDlFiles() { return _dl_files; }
	QString getSelFormat();

public slots:
	void slotDownloadFiles();

protected:
	LogViewerWidget* pv_RemoteLog;
	QHash<QString, QUrl> _dl_files;
	QComboBox* _combo_format;
	QString _format;
};

#endif
