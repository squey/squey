#include <pvkernel/core/PVDirectory.h>
#include <QDir>

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
