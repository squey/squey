#include "include/PVLogViewerDialog.h"


#include <QMenu>
#include <QMenuBar>
#include <QDialogButtonBox>
#include <QWidget>
#include <QVBoxLayout>

PVLogViewerDialog::PVLogViewerDialog(QWidget* parent):
	QDialog(parent)
{
	QMenuBar *rl_menuBar = new QMenuBar(0);
	QMenu *rl_fileMenu = rl_menuBar->addMenu( tr( "Machine" ) );

	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok
					 | QDialogButtonBox::Cancel);

	QVBoxLayout *rl_layout = new QVBoxLayout;

	pv_RemoteLog = new LogViewerWidget();
	rl_fileMenu->addAction(pv_RemoteLog->addMachineAction());
	rl_fileMenu->addAction(pv_RemoteLog->removeMachineAction());

	rl_layout->setMenuBar(rl_menuBar);
	rl_menuBar->show();

	rl_layout->addWidget(pv_RemoteLog);
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
	accept();
}
