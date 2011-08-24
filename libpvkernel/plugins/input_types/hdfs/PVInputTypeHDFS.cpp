#include "PVInputTypeHDFS.h"
#include "../../common/hdfs/PVInputHDFSFile.h"
#include "setenv.h"

#include <QMessageBox>

PVRush::PVInputTypeHDFS::PVInputTypeHDFS():
	PVInputType()
{
	if (!init_env_hadoop()) {
		PVLOG_ERROR("Unable to initialize hadoop environnement. Hadoop support won't work.\n");
	}
}

bool PVRush::PVInputTypeHDFS::createWidget(hash_formats const& /*formats*/, list_inputs &inputs, QString& format, QWidget* parent) const
{
	// TODO: ask for a namenode, list the files of that namenode and chose one or more files !
	
	PVInputHDFSServer_p serv(new PVInputHDFSServer("namenode.hadoop.cluster.picviz", 8020, "hadoop"));
	if (!serv->connect()) {
		QMessageBox err(QMessageBox::Critical, QObject::tr("HDFS server error"), QObject::tr("Unable to connection to HDFS server %1").arg(serv->get_human_name()), QMessageBox::Ok, parent);
		err.exec();
		return false;
	}
	PVInputHDFSFile f(serv, "/data/squid.log");

	QVariant in;
	in.setValue(f);
	inputs.push_back(in);

	format = QString(PICVIZ_AUTOMATIC_FORMAT_STR);

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

QString PVRush::PVInputTypeHDFS::human_name_of_input(PVCore::PVArgument const& in) const
{
	return in.value<PVInputHDFSFile>().get_human_name();
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

bool PVRush::PVInputTypeHDFS::get_custom_formats(PVCore::PVArgument const& in, hash_formats &formats) const
{
	// TODO: find custom format in the hdfs system
	return false;
}

QKeySequence PVRush::PVInputTypeHDFS::menu_shortcut() const
{
	return QKeySequence();
}
