/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVDirectory.h>

#include <cstdlib> // for mkdtemp

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>

// Taken from http://john.nachtimwald.com/2010/06/08/qt-remove-directory-and-its-contents/
bool PVCore::PVDirectory::remove_rec(QString const& dirName)
{
	bool result = true;
	QDir dir(dirName);

	if (dir.exists(dirName)) {
		for (QFileInfo info : dir.entryInfoList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden |
		                                            QDir::AllDirs | QDir::Files,
		                                        QDir::DirsFirst)) {
			if (info.isDir()) {
				result = remove_rec(info.absoluteFilePath());
			} else {
				result = QFile::remove(info.absoluteFilePath());
			}

			if (!result) {
				return result;
			}
		}
		result = dir.rmdir(dirName);
	}

	return result;
}

QString PVCore::PVDirectory::temp_dir(QDir const& directory, QString const& pattern)
{
	QFileInfo fi(pattern);
	QString tmp_dir_pattern = directory.absoluteFilePath(fi.fileName());
	QByteArray tmp_dir_ba = tmp_dir_pattern.toLocal8Bit();
	char* tmp_dir_p = mkdtemp(tmp_dir_ba.data());
	if (tmp_dir_p == nullptr) {
		return QString();
	}
	QString tmp_dir(tmp_dir_p);
	return tmp_dir;
}

QString PVCore::PVDirectory::temp_dir(QString const& pattern)
{
	return temp_dir(QDir::temp(), pattern);
}
