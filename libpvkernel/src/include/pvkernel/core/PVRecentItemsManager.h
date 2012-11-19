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
#include <pvkernel/rush/PVSourceDescription.h>

namespace PVRush
{
class PVInputType;
class PVSourceDescription;
class PVFormat;
}

Q_DECLARE_METATYPE(PVRush::PVSourceDescription)
Q_DECLARE_METATYPE(PVRush::PVFormat)

namespace PVCore
{

/**
 * \class PVRecentItemsManager
 *
 * \note This class is a singleton managing the data recently used by the application.
 */
class PVRecentItemsManager
{
public:
	typedef PVCore::PVSharedPtr<PVRecentItemsManager> PVRecentItemsManager_p;
	typedef QList<QVariant> variant_list_t;

public:
	// List of all available categories of recent items
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

	/*! \brief Return the serializable name of a given category.
	 */
	const QString get_key(Category category)
	{
		return _recents_items_keys[category];
	}

	/*! \brief Add an item (path) for a given category.
	 */
	void add(const QString& item_path, Category category);

	/*! \brief Add a source item for a given category.
	 */
	void add_source(PVRush::PVSourceCreator_p source_creator_p, const PVRush::PVInputType::list_inputs& inputs, const PVRush::PVFormat& format);

	/*! \brief Return the recent items for a given category as a list of QVariant.
	 */
	const variant_list_t get_list(Category category);

private:
	/*! \brief Return a source description from the settings current group.
	 */
	PVRush::PVSourceDescription deserialize_source_description() const;

	/*! \brief Get the best source timestamp to replace (oldest, matching the same source description or 0).
	 *  \return The timestamp to replace or 0 if the list count is less than the maximum allowed items.
	 */
	uint64_t get_source_timestamp_to_replace(const PVRush::PVSourceDescription& source_description);

private:
	/*! \brief Return a list of recent items of a given category as a list of QString QVariant.
	 */
	const variant_list_t items_list(Category category) const;

	/*! \brief Return the recent sources description as a list of QVariant.
	 */
	const variant_list_t sources_description_list() const;

	/*! \brief Return the supported formats as a list of QVariant.
	 */
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
