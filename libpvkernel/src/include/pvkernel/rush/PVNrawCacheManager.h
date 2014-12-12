/**
 * \file PVNRawCacheManager.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVCORE_PVNRAWCACHEMANAGER_H__
#define __PVCORE_PVNRAWCACHEMANAGER_H__

#include <pvkernel/core/PVSharedPointer.h>

#include <memory>

#include <QDirIterator>
#include <QSettings>
class QString;
class QStringList;

namespace PVRush
{

class PVNrawCacheManager
{
public:

	static PVNrawCacheManager& get()
	{
		static PVNrawCacheManager instance;

		return instance;
	}

public:
	static QString nraw_dir();

public:
	void add_investigation(const QString& investigation, const QStringList& nraws);
	void remove_investigation(const QString& investigation, bool remove_from_disk = false);
	void remove_nraws_from_investigation(const QString& investigation);
	void delete_unused_cache();

private:
	QStringList visit_nraw_folders(const QString &base_directory, const QString &name_filter, std::function<bool(QDirIterator& it)> f);
	QStringList list_nraws_used_by_investigations();
	QStringList list_nraws_used_by_investigation(const QString& investigation);

private:
	QString relative_to_absolute_nraw(const QString& relative_nraw) const;
	QStringList relative_to_absolute_nraws(const QStringList& relative_nraws) const;

	QString absolute_to_relative_nraw(const QString& absolute_nraws) const;
	QStringList absolute_to_relative_nraws(const QStringList& absolute_nraws) const;

private:
	void compatibility_move_nraws_to_user_nraws_dir();

private:
	/*! \brief Converts an investigation path to the proper QSettings key.
	 */
	QString path_to_key(const QString& path);

	/*! \brief Converts a QSettings key to its investigation path.
	 */
	QString key_to_path(const QString& key);

private:
	PVNrawCacheManager();
	PVNrawCacheManager(const PVNrawCacheManager&) = delete;
	PVNrawCacheManager& operator=(const PVNrawCacheManager&) = delete;

private:
	mutable std::unique_ptr<QSettings> _cache_file;
};

}

#endif // __PVCORE_PVNRAWCACHEMANAGER_H__
