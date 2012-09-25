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
#include <QDateTime>
#include <QList>
#include <QVariant>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <pvkernel/rush/PVFormat.h>

Q_DECLARE_METATYPE(PVRush::PVSourceDescription)
Q_DECLARE_METATYPE(PVRush::PVFormat)

namespace PVCore
{

class PVRecentItemsManager
{
public:
	typedef PVCore::PVSharedPtr<PVRecentItemsManager> PVRecentItemsManager_p;
	typedef QList<QVariant> variant_list_t;

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

	const QString get_key(Category category)
	{
		return _recents_items_keys[category];
	}

	void add(const QString& item_path, Category category);
	void add_source(PVRush::PVSourceCreator_p source_creator_p, const PVRush::PVInputType::list_inputs& inputs, PVRush::PVFormat& format);
	const variant_list_t get_list(Category category);

private:
	PVRush::PVSourceDescription deserialize_source_description() const;
	uint64_t get_source_timestamp_to_replace(const PVRush::PVSourceDescription& source_description);

private:
	const variant_list_t items_list(Category category) const;
	const variant_list_t sources_description_list() const;
	const variant_list_t supported_format_list() const;

private:
	PVRecentItemsManager() : _recents_settings("recents.ini", QSettings::IniFormat) {}
	PVRecentItemsManager(const PVRecentItemsManager&);
	PVRecentItemsManager &operator=(const PVRecentItemsManager&);

private:
	static PVRecentItemsManager_p _recent_items_manager_p;

	mutable QSettings _recents_settings;
	const int64_t _max_recent_items = 5;
	const QStringList _recents_items_keys = { "recent_projects", "recent_sources", "recent_used_formats", "recent_edited_formats", "supported_formats" };
};

}

#endif /* PVRECENTITEMSMANAGER_H_ */
