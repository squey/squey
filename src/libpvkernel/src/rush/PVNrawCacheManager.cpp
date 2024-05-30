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

#include <pvkernel/rush/PVNraw.h>             // for PVNraw, etc
#include <pvkernel/rush/PVNrawCacheManager.h> // for PVNrawCacheManager

#include <pvkernel/core/PVConfig.h>     // for PVConfig
#include <pvkernel/core/PVDirectory.h>  // for remove_rec
#include <pvkernel/core/PVFileHelper.h> // for PVFileHelper

#include <pvbase/general.h> // for SQUEY_PATH_SEPARATOR_CHAR

#include <QByteArray> // for QByteArray
#include <QDir>       // for QDir, etc
#include <QDirIterator>
#include <QFile> // for QFile
#include <QFileInfo>
#include <QSettings> // for QSettings, etc
#include <QString>
#include <QStringList>

#include <functional> // for function
#include <memory>     // for unique_ptr

PVRush::PVNrawCacheManager::PVNrawCacheManager()
    : _cache_file(
          new QSettings(QString(PVCore::PVConfig::get()
                                    .config()
                                    .value(QString::fromStdString(PVRush::PVNraw::config_nraw_tmp),
                                           QString::fromStdString(PVRush::PVNraw::default_tmp_path))
                                    .toString()) +
                            SQUEY_PATH_SEPARATOR_CHAR + ".cache",
                        QSettings::IniFormat))
{
	_cache_file->beginGroup("cache");

	compatibility_move_nraws_to_user_nraws_dir();
}

void PVRush::PVNrawCacheManager::add_investigation(const QString& investigation,
                                                   const QStringList& absolute_nraws)
{
	_cache_file->setValue(investigation, absolute_to_relative_nraws(absolute_nraws));
}

void PVRush::PVNrawCacheManager::remove_investigation(const QString& investigation,
                                                      bool remove_from_disk /* = false */)
{
	// Delete Nraw folders
	remove_nraws_from_investigation(investigation);

	// Remove entry from cache file
	_cache_file->remove(investigation.right(investigation.length() - 1));

	// Delete investigation file
	if (remove_from_disk) {
		QFile::remove(investigation);
	}
}

void PVRush::PVNrawCacheManager::remove_nraws_from_investigation(const QString& investigation)
{
	for (const QString& dir : list_nraws_used_by_investigation(investigation)) {
		PVCore::PVDirectory::remove_rec(dir);
	}
}

void PVRush::PVNrawCacheManager::delete_unused_cache()
{
	QString regexp = QString::fromStdString(PVRush::PVNraw::nraw_tmp_name_regexp);

	QStringList nraw_without_opened_files =
	    visit_nraw_folders(nraw_dir(), regexp, [](QDirIterator& it) {
		    bool has_opened_file = false;

		    while (it.hasNext()) {
			    it.next();
			    QByteArray data = it.filePath().toLocal8Bit();
			    const char* c_file_name = data.data();
			    has_opened_file |= PVCore::PVFileHelper::is_already_opened(c_file_name);
		    }

		    return !has_opened_file;
	    });
	nraw_without_opened_files.sort();

	QStringList nraws_used_by_investigation = list_nraws_used_by_investigations();
	nraws_used_by_investigation.sort();

	for (const QString& nraw : nraws_used_by_investigation) {
		nraw_without_opened_files.removeAll(nraw);
	}

	for (const QString& dir : nraw_without_opened_files) {
		PVCore::PVDirectory::remove_rec(dir);
	}
}

QStringList PVRush::PVNrawCacheManager::list_nraws_used_by_investigations()
{
	QStringList investigations = _cache_file->allKeys();
	QStringList used_nraws;
	for (const QString& investigation : investigations) {

		QStringList nraws = _cache_file->value(investigation).toStringList();
		used_nraws << nraws;

		// Remove cache line if investigation doesn't exist anymore
		QString path = key_to_path(investigation);
		if (!QFile(path).exists()) {
			_cache_file->remove(path);
		}
	}
	return relative_to_absolute_nraws(used_nraws);
}

QString PVRush::PVNrawCacheManager::nraw_dir()
{
	return PVCore::PVConfig::get()
	           .config()
	           .value(QString::fromStdString(PVRush::PVNraw::config_nraw_tmp),
	                  QString::fromStdString(PVRush::PVNraw::default_tmp_path))
	           .toString() +
	       QDir::separator() + PVCore::PVConfig::username();
}

QStringList
PVRush::PVNrawCacheManager::list_nraws_used_by_investigation(const QString& investigation)
{
	return _cache_file->value(path_to_key(investigation)).toStringList();
}

QStringList PVRush::PVNrawCacheManager::visit_nraw_folders(const QString& base_directory,
                                                           const QString& name_filter,
                                                           std::function<bool(QDirIterator& it)> f)
{
	QDir nraw_dir_base(base_directory, name_filter);
	QStringList result;

	for (const QString& sub_dir_name : nraw_dir_base.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
		QString sub_dir_path(base_directory + QDir::separator() + sub_dir_name);
		QDirIterator it(sub_dir_path, QDir::Files | QDir::NoDotAndDotDot,
		                QDirIterator::Subdirectories);
		if (f(it)) {
			result << base_directory + QDir::separator() + sub_dir_name;
		}
	}

	return result;
}

QString PVRush::PVNrawCacheManager::relative_to_absolute_nraw(const QString& relative_nraw) const
{
	return nraw_dir() + QDir::separator() + relative_nraw;
}

QStringList
PVRush::PVNrawCacheManager::relative_to_absolute_nraws(const QStringList& relative_nraws) const
{
	QStringList absolute_nraws;

	for (const QString& relative_nraw : relative_nraws) {
		absolute_nraws << relative_to_absolute_nraw(relative_nraw);
	}

	return absolute_nraws;
}

QString PVRush::PVNrawCacheManager::absolute_to_relative_nraw(const QString& absolute_nraw) const
{
	return QDir(absolute_nraw).dirName();
}

QStringList
PVRush::PVNrawCacheManager::absolute_to_relative_nraws(const QStringList& absolute_nraws) const
{
	QStringList relative_nraws;

	for (const QString& absolute_nraw : absolute_nraws) {
		relative_nraws << absolute_to_relative_nraw(absolute_nraw);
	}

	return relative_nraws;
}

QString PVRush::PVNrawCacheManager::path_to_key(const QString& path)
{
	return path.right(path.length() - 1);
}

QString PVRush::PVNrawCacheManager::key_to_path(const QString& key)
{
	return QString(SQUEY_PATH_SEPARATOR_CHAR) + key;
}

void PVRush::PVNrawCacheManager::compatibility_move_nraws_to_user_nraws_dir()
{
	// Compatibility : Move nraw temp directories to nraw user subdirectory
	QString nraw_dir_base = PVCore::PVConfig::get()
	                            .config()
	                            .value(QString::fromStdString(PVRush::PVNraw::config_nraw_tmp),
	                                   QString::fromStdString(PVRush::PVNraw::default_tmp_path))
	                            .toString();
	QString user_nraw_dir_base = nraw_dir_base + QDir::separator() + PVCore::PVConfig::username();
	QFileInfo fi(user_nraw_dir_base);
	if (fi.exists() == false) {
		fi.dir().mkpath(user_nraw_dir_base);

		QDir nraw_dirs(nraw_dir_base, QString::fromStdString(PVRush::PVNraw::nraw_tmp_name_regexp));
		for (const QString& sub_dir_name : nraw_dirs.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
			QFile::rename(nraw_dir_base + QDir::separator() + sub_dir_name,
			              user_nraw_dir_base + QDir::separator() + sub_dir_name);
		}

		// Update cache file entries
		QStringList investigations = _cache_file->allKeys();
		QStringList used_nraws;
		for (const QString& investigation : investigations) {
			QString nraw_temp_folder = _cache_file->value(investigation).toString();

			_cache_file->setValue(investigation, absolute_to_relative_nraw(nraw_temp_folder));
		}

		// Move cache file
		QString new_cache_file_path =
		    user_nraw_dir_base + QDir::separator() + QFileInfo(_cache_file->fileName()).fileName();
		_cache_file->sync();
		QFile::rename(_cache_file->fileName(), new_cache_file_path);
		_cache_file = std::make_unique<QSettings>(
		    new_cache_file_path, QSettings::IniFormat); // reload cache file
	}
}
