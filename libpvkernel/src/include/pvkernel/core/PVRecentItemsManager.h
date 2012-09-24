/**
 * \file PVRecentItemsManager.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVRECENTITEMSMANAGER_H_
#define PVRECENTITEMSMANAGER_H_

#include <QObject>
#include <QSettings>
#include <QStringList>

#include <pvkernel/core/PVSharedPointer.h>


namespace PVCore
{

class PVRecentItemsManager
{
public:
	typedef PVCore::PVSharedPtr<PVRecentItemsManager> PVRecentItemsManager_p;

public:
	enum Category {
		FIRST = 0,

		PROJECTS = FIRST,
		SOURCES,
		USED_FORMATS,
		EDITED_FORMATS,
		SUPPORTED_FORMATS,

		LAST
	};

	static PVRecentItemsManager_p& get()
	{
		if (_recent_items_manager_p.get() == nullptr) {
			_recent_items_manager_p = PVRecentItemsManager_p(new PVRecentItemsManager());
		}
		return _recent_items_manager_p;
	}

	void add(const QString& item_path, Category category)
	{
		QString recent_items_key = _recents_items_keys[category];
	    QStringList files = _recents_settings.value(recent_items_key).toStringList();
	    files.removeAll(item_path);
	    files.prepend(item_path);

	    for (; files.size() > _max_recent_items; files.removeLast()) {}

	    _recents_settings.setValue(recent_items_key, files);
	}

	const QStringList get_list(Category category)
	{
		if (category == Category::SUPPORTED_FORMATS) {
			return get_supported_formats_paths();
		}
		else {
			return _recents_settings.value(_recents_items_keys[category]).toStringList();
		}
	}

	const QString get_key(Category category)
	{
		return _recents_items_keys[category];
	}

	QSettings& get_qsettings() { return _recents_settings; }

/*signals:
	void recent_items_updated(int category);*/

private:
	const QStringList get_supported_formats_paths()
	{
		QStringList support_formats;

		/*LIB_CLASS(PVRush::PVInputType) &input_types = LIB_CLASS(PVRush::PVInputType)::get();
		LIB_CLASS(PVRush::PVInputType)::list_classes const& lf = input_types.get_list();

		LIB_CLASS(PVRush::PVInputType)::list_classes::const_iterator it;

		for (it = lf.begin(); it != lf.end(); it++) {
			PVRush::PVInputType_p in = it.value();

			PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in);
			PVRush::hash_format_creator format_creator = PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

			PVRush::hash_format_creator::const_iterator itfc;
			for (itfc = format_creator.begin(); itfc != format_creator.end(); itfc++) {
				support_formats << itfc.value().first.get_full_path();
			}
		}*/

		return support_formats;
	}

private:
	PVRecentItemsManager() : _recents_settings("recents.ini", QSettings::IniFormat) {}
	PVRecentItemsManager(const PVRecentItemsManager&);
	PVRecentItemsManager &operator=(const PVRecentItemsManager&);

private:
	static PVRecentItemsManager_p _recent_items_manager_p;
	//static PVRecentItemsManager* _recent_items_manager;

	QSettings _recents_settings;
	const int64_t _max_recent_items = 5;
	const QStringList _recents_items_keys = { "recent_projects", "recent_sources", "recent_used_formats", "recent_edited_formats", "supported_formats" };
};

}

#endif /* PVRECENTITEMSMANAGER_H_ */
