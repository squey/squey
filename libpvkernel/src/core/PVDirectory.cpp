#include <pvkernel/core/PVDirectory.h>
#include <QDir>
#include <QByteArray>

#include <stdlib.h>

#ifdef WIN32
#include <io.h>
#include <string.h>
static char* mkdtemp(char* pattern)
{
	errno_t res = _mktemp_s(pattern, strlen(pattern)+1);
	if (res == 0) {
		return pattern;
	}

	return NULL;
}
#endif

// Taken from http://john.nachtimwald.com/2010/06/08/qt-remove-directory-and-its-contents/
bool PVCore::PVDirectory::remove_rec(QString const& dirName)
{
	bool result = true;
	QDir dir(dirName);

	if (dir.exists(dirName)) {
		Q_FOREACH(QFileInfo info, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden  | QDir::AllDirs | QDir::Files, QDir::DirsFirst)) {
			if (info.isDir()) {
				result = remove_rec(info.absoluteFilePath());
			}
			else {
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

QString PVCore::PVDirectory::temp_dir(QString const& pattern)
{
	QString tmp_dir_pattern = QDir::temp().absoluteFilePath(pattern);
	QByteArray tmp_dir_ba = tmp_dir_pattern.toLocal8Bit();
	char* tmp_dir_p = mkdtemp(tmp_dir_ba.data());
	if (tmp_dir_p == NULL) {
		return QString();
	}
	QString tmp_dir(tmp_dir_p);
	return tmp_dir;
}
