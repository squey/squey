//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

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
	return temp_dir(PVRush::PVNrawCacheManager::nraw_dir(), pattern);
}
