/**
 * \file PVNRawCacheManager.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVCORE_PVNRAWCACHEMANAGER_H__
#define __PVCORE_PVNRAWCACHEMANAGER_H__

#include <pvkernel/core/PVSharedPointer.h>

#include <QDirIterator>
#include <QSettings>
class QString;
class QStringList;

namespace PVRush
{

class PVNrawCacheManager
{
public:
	typedef PVCore::PVSharedPtr<PVNrawCacheManager> PVNrawCacheManager_p;

	static PVNrawCacheManager_p& get()
	{
		if (_cache_manager_p.get() == nullptr) {
			_cache_manager_p = PVNrawCacheManager_p(new PVNrawCacheManager());
		}
		return _cache_manager_p;
	}

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
	static PVNrawCacheManager_p _cache_manager_p;
	mutable QSettings _cache_file;
};

}

#endif // __PVCORE_PVNRAWCACHEMANAGER_H__
