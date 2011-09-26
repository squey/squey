#include "PVInputTypeRemoteFilename.h"
#include "include/PVLogViewerDialog.h"

#include <QMessageBox>
#include <QFileInfo>

#include <stdlib.h>

#ifndef WIN32
#include <sys/time.h>
#include <sys/resource.h>
#endif

PVRush::PVInputTypeRemoteFilename::PVInputTypeRemoteFilename() :
	PVInputTypeFilename()
{
}

bool PVRush::PVInputTypeRemoteFilename::createWidget(hash_formats const& formats, list_inputs &inputs, QString& format, QWidget* parent) const
{
	PVLogViewerDialog *RemoteLogDialog = new PVLogViewerDialog(parent);
	if (RemoteLogDialog->exec() == QDialog::Rejected) {
		RemoteLogDialog->deleteLater();
		return false;
	}
	// Force deletion so that settings are saved
	RemoteLogDialog->deleteLater();

	format = PICVIZ_AUTOMATIC_FORMAT_STR;
	QStringList const& files = RemoteLogDialog->getDlFiles();
	for (int i = 0; i < files.size(); i++) {
		PVLOG_INFO("%s\n", qPrintable(files[i]));
	}
	return load_files(files, true, inputs, parent);
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

bool PVRush::PVInputTypeRemoteFilename::get_custom_formats(PVCore::PVArgument const& in, hash_formats &formats) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeRemoteFilename::menu_shortcut() const
{
	return QKeySequence();
}
