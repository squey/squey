/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRECENTITEMSMANAGER_H_
#define PVRECENTITEMSMANAGER_H_

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceDescription.h>

#include <QSettings>
#include <QStringList>
#include <QList>

#include <sigc++/sigc++.h>

namespace PVRush
{
class PVFormat;
} // namespace PVRush

Q_DECLARE_METATYPE(PVRush::PVSourceDescription)
Q_DECLARE_METATYPE(PVRush::PVFormat)

namespace PVCore
{

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

template <Category c>
struct list_type {
};

template <>
struct list_type<USED_FORMATS> {
	using type = QStringList;
};
template <>
struct list_type<EDITED_FORMATS> {
	using type = QStringList;
};
template <>
struct list_type<PROJECTS> {
	using type = QStringList;
};
template <>
struct list_type<SOURCES> {
	using type = QList<PVRush::PVSourceDescription>;
};
template <>
struct list_type<SUPPORTED_FORMATS> {
	using type = QStringList;
};

/**
 * \class PVRecentItemsManager
 *
 * \note This class is a singleton managing the data recently used by the application.
 */
class PVRecentItemsManager
{
  public:
	static PVRecentItemsManager& get()
	{
		static PVRecentItemsManager recent_items_manager;
		return recent_items_manager;
	}

	/*! \brief Return the serializable name of a given category.
	 */
	const QString get_key(Category category) { return _recents_items_keys[category]; }

	/*! \brief Add an item (path) for a given category.
	 */
	template <Category category>
	void add(const QString& item_path)
	{
		QString recent_items_key = _recents_items_keys[category];
		QStringList files = _recents_settings.value(recent_items_key).toStringList();
		files.removeAll(item_path);
		files.prepend(item_path);
		if (category != Category::PROJECTS) {
			while (files.size() > _max_recent_items) {
				files.removeLast();
			}
		}
		_recents_settings.setValue(recent_items_key, files);
		_recents_settings.sync();
		_add_item.emit(category);
	}

	/*! \brief Add a source item for a given category.
	 */
	void add_source(PVRush::PVSourceCreator_p source_creator_p,
	                const PVRush::PVInputType::list_inputs& inputs,
	                const PVRush::PVFormat& format);

	// Helper function to call a generic lambda on every category at compile time.
	// Could be improve with generic lambda (cxx14) for now, we use only functors ...

	template <class F, Category... c>
	static void __apply_on_category(F&& f)
	{
		int _[] __attribute__((unused)) = {(f.template call<c>(), 1)...};
	}

	template <class F, Category... c>
	static void apply_on_category(F&& f)
	{
		__apply_on_category<F, PROJECTS, SOURCES, USED_FORMATS, EDITED_FORMATS, SUPPORTED_FORMATS>(
		    std::forward<F>(f));
	}

	/*! \brief Return a source description from the settings current group.
	 */
	template <Category category>
	typename list_type<category>::type get_list();

	void clear(Category category, QList<int> indexes = QList<int>());

	std::tuple<QString, QStringList> get_string_from_entry(QString const& string) const;
	std::tuple<QString, QStringList>
	get_string_from_entry(PVRush::PVSourceDescription const& sd) const;
	std::tuple<QString, QStringList> get_string_from_entry(PVRush::PVFormat const& f) const;

  private:
	PVRush::PVSourceDescription deserialize_source_description();

	/*! \brief Get the best source timestamp to replace (oldest, matching the same source
	 * description or 0).
	 *  \return The timestamp to replace or 0 if the list count is less than the maximum allowed
	 * items.
	 */
	uint64_t get_source_timestamp_to_replace(const PVRush::PVSourceDescription& source_description);

	void clear_missing_files();

  private:
	/*! \brief Return a list of recent items of a given category as a list of QString.
	 */
	QStringList items_list(Category category) const;

	/**
	 * Remove value in recent file when pointed file is missing.
	 */
	void remove_missing_files(Category category);

	/*! \brief Return the recent sources description as a list
	 */
	// FIXME : This function is not const as it required group to list sources and Qt doesn't
	// provide this interface
	QList<PVRush::PVSourceDescription> sources_description_list();
	void remove_invalid_source();

	/*! \brief Return the supported formats as a list
	 */
	QStringList supported_format_list() const;

  private:
	PVRecentItemsManager();
	PVRecentItemsManager(const PVRecentItemsManager&);
	PVRecentItemsManager& operator=(const PVRecentItemsManager&);

  public:
	sigc::signal<void, Category> _add_item;

  private:
	QSettings _recents_settings;
	const int64_t _max_recent_items = 30;
	const QStringList _recents_items_keys = {"recent_projects", "recent_sources",
	                                         "recent_used_formats", "recent_edited_formats",
	                                         "supported_formats"};
};
} // namespace PVCore

#endif /* PVRECENTITEMSMANAGER_H_ */
