/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#include <pvguiqt/PVChangelogMessage.h>
#include <pvguiqt/PVAboutBoxDialog.h>

#include <pvkernel/core/PVConfig.h>

#include <QDir>
#include <QFile>
#include <QTextStream>

#include INENDI_VERSION_FILE_PATH

PVGuiQt::PVChangelogMessage::PVChangelogMessage(QWidget* parent /* = nullptr*/)
{
	QString current_version = QString(INENDI_CURRENT_VERSION_STR);
	QString previous_version;

	QFile version_file(QString::fromStdString(PVCore::PVConfig::user_dir()) + QDir::separator() +
	                   "version.txt");
	if (version_file.open(QFile::ReadOnly | QFile::Text)) {
		QTextStream in(&version_file);
		previous_version = in.readLine();
		version_file.close();
	}

	if (current_version != previous_version) {
		PVGuiQt::PVAboutBoxDialog aboutbox(parent);
		aboutbox.select_changelog_tab();
		aboutbox.exec();
	}

	if (version_file.open(QIODevice::WriteOnly)) {
		QTextStream out(&version_file);
		out << current_version;
	}
}
