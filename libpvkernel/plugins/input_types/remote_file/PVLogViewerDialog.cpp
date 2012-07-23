/**
 * \file PVLogViewerDialog.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "include/PVLogViewerDialog.h"


#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QDialogButtonBox>
#include <QWidget>
#include <QVBoxLayout>
#include <QSpacerItem>

PVLogViewerDialog::PVLogViewerDialog(QStringList const& formats, QWidget* parent):
	QDialog(parent)
{
	QMenuBar *rl_menuBar = new QMenuBar(0);
	QMenu *rl_fileMenu = rl_menuBar->addMenu( tr( "Machine" ) );

	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok
					 | QDialogButtonBox::Cancel);

	QVBoxLayout *rl_layout = new QVBoxLayout;

	pv_RemoteLog = new LogViewerWidget(this);
	rl_fileMenu->addAction(pv_RemoteLog->addMachineAction());
	rl_fileMenu->addAction(pv_RemoteLog->removeMachineAction());

	rl_layout->setMenuBar(rl_menuBar);
	rl_menuBar->show();

	rl_layout->addWidget(pv_RemoteLog);
	
	QHBoxLayout* formatLayout = new QHBoxLayout();
	formatLayout->addWidget(new QLabel(tr("Format:")));
	_combo_format = new QComboBox();
	_combo_format->addItems(formats);
	formatLayout->addWidget(_combo_format);
	formatLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));

	rl_layout->addLayout(formatLayout);
	rl_layout->addWidget(buttonBox);

	connect(buttonBox, SIGNAL(accepted()), this, SLOT(slotDownloadFiles()));
	connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

	setLayout(rl_layout);
	setWindowTitle(tr("Import remote file"));
}

PVLogViewerDialog::~PVLogViewerDialog()
{
	pv_RemoteLog->deleteLater();
}

void PVLogViewerDialog::slotDownloadFiles()
{
	QHash<QString, QUrl> tmp_files;
	if (!pv_RemoteLog->downloadSelectedFiles(tmp_files)) {
		return;
	}

	_dl_files = tmp_files;
	_format = _combo_format->currentText();
	accept();
}

QString PVLogViewerDialog::getSelFormat()
{
	return _format;
}
