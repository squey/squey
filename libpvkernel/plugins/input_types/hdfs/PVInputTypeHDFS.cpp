/**
 * \file PVInputTypeHDFS.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVImportHDFSDlg.h"
#include "PVInputTypeHDFS.h"
#include "setenv.h"

#include <QMessageBox>

PVRush::PVInputTypeHDFS::PVInputTypeHDFS():
	PVInputTypeDesc<PVInputHDFSFile>()
{
	if (!init_env_hadoop()) {
		PVLOG_ERROR("Unable to initialize hadoop environnement. Hadoop support won't work.\n");
	}
}

bool PVRush::PVInputTypeHDFS::createWidget(hash_formats const& formats, hash_formats& /*new_formats*/, list_inputs &inputs, QString& format, PVCore::PVArgumentList& args_ext, QWidget* parent) const
{
	// TODO: ask for a namenode, list the files of that namenode and chose one or more files !
	QStringList formats_name = formats.keys();

	formats_name.prepend(QString(PICVIZ_AUTOMATIC_FORMAT_STR));
	formats_name.prepend(QString(PICVIZ_BROWSE_FORMAT_STR));

	PVImportHDFSDlg* dlg = new PVImportHDFSDlg(formats_name, parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	PVInputHDFSServer_p serv(new PVInputHDFSServer(dlg->namenode(), dlg->port(), dlg->user()));
	/*
	if (!serv->connect()) {
		QMessageBox err(QMessageBox::Critical, QObject::tr("HDFS server error"), QObject::tr("Unable to connection to HDFS server %1").arg(serv->get_human_name()), QMessageBox::Ok, parent);
		err.exec();
		return false;
	}
	*/
	PVInputDescription_p f(new PVInputHDFSFile(serv, dlg->path()));
	inputs.push_back(f);

	format = dlg->format_name();

	return inputs.size() > 0;
}

PVRush::PVInputTypeHDFS::~PVInputTypeHDFS()
{
}


QString PVRush::PVInputTypeHDFS::name() const
{
	return QString("hdfs");
}

QString PVRush::PVInputTypeHDFS::human_name() const
{
	return QString("HDFS import plugin");
}

QString PVRush::PVInputTypeHDFS::human_name_serialize() const
{
	return tr("HDFS files");
}

QString PVRush::PVInputTypeHDFS::menu_input_name() const
{
	return QString("Import from HDFS...");
}

QString PVRush::PVInputTypeHDFS::tab_name_of_inputs(list_inputs const& in) const
{
	// TODO!!
	return QString("TODO: tab name for hdfs");
}

bool PVRush::PVInputTypeHDFS::get_custom_formats(PVInputDescription_p in, hash_formats &formats) const
{
	// TODO: find custom format in the hdfs system
	return false;
}

QKeySequence PVRush::PVInputTypeHDFS::menu_shortcut() const
{
	return QKeySequence();
}
