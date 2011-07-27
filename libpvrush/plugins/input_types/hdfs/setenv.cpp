// AG: this file is used to set the CLASSPATH for libhdfs to work.
// It will look for the hadoop installation directory first in the config file,
// and then in the $HADOOP_HOME environnement variable

#include <pvcore/general.h> // For pvconfig

#include <QString>
#include <QStringList>
#include <QDir>

#include <stdlib.h>
#include <string.h>
#include <errno.h>

static void add_classpath(QString& classpath, QFileInfoList jars)
{
	// Add the jars in 'jars' into the classpath
	
	// Add quotes between the path to the jars
	QStringList jars_path;
	for (int i = 0; i < jars.size(); i++) {
		QString path = jars[i].absoluteFilePath();
		//jars_path << QString("\"") + path + QString("\"");
		jars_path << path;
	}
	classpath += jars_path.join(":");
}

static bool process_dir(QString& classpath, QString path_dir)
{
	QDir dir(path_dir);
	QFileInfoList jars = dir.entryInfoList(QStringList() << "*.jar", QDir::Files | QDir::Readable);
	if (jars.size() == 0) {
		PVLOG_ERROR("(hadoop-init) unable to find jar files in %s. Hadoop support won't work.\n", qPrintable(path_dir));
		return false;
	}
	add_classpath(classpath, jars);
	return true;
}

QString find_hadoop_path()
{
	QString env;
	char* hadoop_env = getenv("HADOOP_HOME");
	if (hadoop_env) {
		env = hadoop_env;
	}

	QString conf = pvconfig.value("hadoop/hadoop_home").toString();

	return (conf.isEmpty()) ? env : conf;
}

bool init_env_hadoop()
{
#ifdef WIN32
	PVLOG_ERROR("Hadoop support is only under Linux for now !\n");
	return false;
#endif

	QString hadoop_path = find_hadoop_path();
	if (hadoop_path.isEmpty()) {
		return false;
	}


	// Put all the jars in $HADOOP_HOME/*.jar and $HADOOP_HOME/lib/*.jar in the CLASSPATH
	// This is UNIX only for now
	QString classpath;
	if (!process_dir(classpath, hadoop_path) || !process_dir(classpath, hadoop_path + QString("/lib"))) {
		return false;
	}

	// Now, set the environnement variable
	if (setenv("CLASSPATH", qPrintable(classpath), 1) != 0) {
		PVLOG_ERROR("Unable to set the classpath environnement: %s\n", strerror(errno));
		return false;
	}

	PVLOG_DEBUG("(hadoop-init) CLASSPATH has been set to '%s'\n", qPrintable(classpath));

	return true;
}
