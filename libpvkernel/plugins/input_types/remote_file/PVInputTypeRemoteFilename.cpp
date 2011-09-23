#include "PVInputTypeRemoteFilename.h"
#include "include/logviewerwidget.h"

#include <QMessageBox>
#include <QFileInfo>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QDialogButtonBox>
#include <QWidget>
#include <QVBoxLayout>

#include <stdlib.h>

#ifndef WIN32
#include <sys/time.h>
#include <sys/resource.h>
#endif

PVRush::PVInputTypeRemoteFilename::PVInputTypeRemoteFilename() :
	PVInputType()
{
}

bool PVRush::PVInputTypeRemoteFilename::createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent) const
{
	QMainWindow *RemoteLogDialog = new QMainWindow(parent);
	QMenuBar *rl_menuBar = new QMenuBar(0);
	QMenu *rl_fileMenu = rl_menuBar->addMenu( tr( "Machine" ) );

	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok
					 | QDialogButtonBox::Cancel);

	QWidget *rl_central = new QWidget();
	QVBoxLayout *rl_layout = new QVBoxLayout;

	LogViewerWidget* pv_RemoteLog = new LogViewerWidget();
	rl_fileMenu->addAction(pv_RemoteLog->addMachineAction());
	rl_fileMenu->addAction(pv_RemoteLog->removeMachineAction());

	RemoteLogDialog->setMenuBar(rl_menuBar);
	rl_menuBar->show();

	rl_central->setLayout(rl_layout);
	rl_layout->addWidget(pv_RemoteLog);
	rl_layout->addWidget(buttonBox);

	connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
	connect(buttonBox, SIGNAL(rejected()), RemoteLogDialog, SLOT(hide()));

	RemoteLogDialog->setWindowTitle(tr("Import remote file"));
	RemoteLogDialog->setCentralWidget(rl_central);

	RemoteLogDialog->show();

	return false;
}

PVRush::PVInputTypeRemoteFilename::~PVInputTypeRemoteFilename()
{
}


QString PVRush::PVInputTypeRemoteFilename::name() const
{
	return QString("remote_file");
}

QString PVRush::PVInputTypeRemoteFilename::human_name() const
{
	return QString("Remote file import plugin");
}

QString PVRush::PVInputTypeRemoteFilename::human_name_of_input(PVCore::PVArgument const& in) const
{
	return in.toString();
}

QString PVRush::PVInputTypeRemoteFilename::menu_input_name() const
{
	return QString("Import remote files...");
}

QString PVRush::PVInputTypeRemoteFilename::tab_name_of_inputs(list_inputs const& in) const
{
	QString tab_name;
	QFileInfo fi(in[0].toString());
	if (in.count() == 1) {
		tab_name = fi.fileName();
	}
	else {
		tab_name = fi.canonicalPath();
	}
	return tab_name;
}

bool PVRush::PVInputTypeRemoteFilename::get_custom_formats(PVCore::PVArgument const& in, hash_formats &formats) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeRemoteFilename::menu_shortcut() const
{
	return QKeySequence();
}
