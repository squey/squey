/**
 * \file PVNrawCacheManager.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/core/PVFileHelper.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVConfig.h>

#include <QFileInfo>
#include <QDirIterator>
#include <QString>
#include <QStringList>

PVRush::PVNrawCacheManager::PVNrawCacheManager() :
	_cache_file(std::unique_ptr<QSettings>(new QSettings(QString(PVCore::PVConfig::get().config().value(PVRush::PVNraw::config_nraw_tmp, PVRush::PVNraw::default_tmp_path).toString()) \
			    + PICVIZ_PATH_SEPARATOR_CHAR + ".cache", QSettings::IniFormat)))
{
	_cache_file->beginGroup("cache");

	compatibility_move_nraws_to_user_nraws_dir();
}

void PVRush::PVNrawCacheManager::add_investigation(
	const QString& investigation,
	const QStringList& absolute_nraws)
{
	_cache_file->setValue(investigation, absolute_to_relative_nraws(absolute_nraws));
}

void PVRush::PVNrawCacheManager::remove_investigation(const QString& investigation, bool remove_from_disk /* = false */)
{
	// Delete Nraw folders
	remove_nraws_from_investigation(investigation);

	// Remove entry from cache file
	_cache_file->remove(investigation.right(investigation.length()-1));

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
	QSettings &pvconfig = PVCore::PVConfig::get().config();

	QString regexp(PVRush::PVNraw::nraw_tmp_name_regexp);

	QStringList nraw_without_opened_files = visit_nraw_folders(nraw_dir(), regexp, [](QDirIterator& it) {
		bool has_opened_file = false;

		while(it.hasNext()) {
			if (it.fileName().isEmpty()) {
				it.next();
				continue;
			}

			const char *c_file_name = it.filePath().toLocal8Bit().data();
			has_opened_file |= PVCore::PVFileHelper::is_already_opened(c_file_name);

			it.next();
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
		if(!QFile(path).exists()) {
			_cache_file->remove(path);
		}
	}
	return relative_to_absolute_nraws(used_nraws);
}

QString PVRush::PVNrawCacheManager::nraw_dir()
{
	return PVCore::PVConfig::get().config().value(PVRush::PVNraw::config_nraw_tmp).toString() + QDir::separator() + PVCore::PVConfig::username();
}

QStringList PVRush::PVNrawCacheManager::list_nraws_used_by_investigation(const QString& investigation)
{
	return _cache_file->value(path_to_key(investigation)).toStringList();
}

QStringList PVRush::PVNrawCacheManager::visit_nraw_folders(
	const QString &base_directory,
    const QString &name_filter,
    std::function<bool(QDirIterator& it)> f
)
{
	QDir nraw_dir_base(base_directory, name_filter);
	QStringList result;

	for (const QString &sub_dir_name : nraw_dir_base.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
		QString sub_dir_path(base_directory + QDir::separator() + sub_dir_name);
		QDirIterator it(sub_dir_path,
		                QDir::Files | QDir::NoDotAndDotDot,
		                QDirIterator::Subdirectories);

		if(f(it)) {
			result << base_directory + QDir::separator() + sub_dir_name;
		}
	}

	return result;
}

QString PVRush::PVNrawCacheManager::relative_to_absolute_nraw(const QString& relative_nraw) const
{
	return nraw_dir() + QDir::separator() + relative_nraw;
}

QStringList PVRush::PVNrawCacheManager::relative_to_absolute_nraws(const QStringList& relative_nraws) const
{
	QStringList absolute_nraws;

	for (const QString& relative_nraw : relative_nraws) {
		absolute_nraws << relative_to_absolute_nraw(relative_nraw);
	}

	return absolute_nraws;
}

QString PVRush::PVNrawCacheManager::absolute_to_relative_nraw(const QString& absolute_nraw) const
{
	return QFileInfo(absolute_nraw).fileName();
}

QStringList PVRush::PVNrawCacheManager::absolute_to_relative_nraws(const QStringList& absolute_nraws) const
{
	QStringList relative_nraws;

	for (const QString& absolute_nraw : absolute_nraws) {
		relative_nraws << absolute_to_relative_nraw(absolute_nraw);
	}

	return relative_nraws;
}

QString PVRush::PVNrawCacheManager::path_to_key(const QString& path)
{
	return path.right(path.length()-1);
}

QString PVRush::PVNrawCacheManager::key_to_path(const QString& key)
{
	return QString(PICVIZ_PATH_SEPARATOR_CHAR) + key;
}

void PVRush::PVNrawCacheManager::compatibility_move_nraws_to_user_nraws_dir()
{
	// Compatibility : Move nraw temp directories to nraw user subdirectory
	QString nraw_dir_base = PVCore::PVConfig::get().config().value(PVRush::PVNraw::config_nraw_tmp).toString();
	QString user_nraw_dir_base = nraw_dir_base + QDir::separator() + PVCore::PVConfig::username();
	QFileInfo fi(user_nraw_dir_base);
	if (fi.exists() == false) {
		fi.dir().mkpath(user_nraw_dir_base);

		QDir nraw_dirs(nraw_dir_base, PVRush::PVNraw::nraw_tmp_name_regexp);
		for (const QString &sub_dir_name : nraw_dirs.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
			QFile::rename(nraw_dir_base + QDir::separator() + sub_dir_name, user_nraw_dir_base + QDir::separator() + sub_dir_name);
		}

		// Update cache file entries
		QStringList investigations = _cache_file->allKeys();
		QStringList used_nraws;
		for (const QString& investigation : investigations) {
			QString nraw_temp_folder = _cache_file->value(investigation).toString();

			_cache_file->setValue(investigation, absolute_to_relative_nraw(nraw_temp_folder));
		}

		// Move cache file
		QString new_cache_file_path = user_nraw_dir_base + QDir::separator() + QFileInfo(_cache_file->fileName()).fileName();
		_cache_file->sync();
		QFile::rename(_cache_file->fileName(), new_cache_file_path);
		_cache_file.reset(new QSettings(new_cache_file_path, QSettings::IniFormat)); // reload cache file
	}
}

