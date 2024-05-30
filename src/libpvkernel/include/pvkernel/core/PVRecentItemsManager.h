/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

namespace PVCore
{

// List of all available categories of recent items
enum Category {
	FIRST = 0,

	PROJECTS = FIRST,
	SOURCES,
	USED_FORMATS,
	EDITED_FORMATS,

	LAST
};

template <Category c>
struct list_type {
	using type = std::vector<std::string>;
};

template <>
struct list_type<SOURCES> {
	using type = std::vector<PVCore::PVSerializedSource>;
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
		_add_item[category].emit();
	}

	/*! \brief Add a source item for a given category.
	 */
	void add_source(PVRush::PVSourceCreator_p source_creator_p,
	                const PVRush::PVInputType::list_inputs& inputs,
	                const PVRush::PVFormat& format);

	template <class F, Category... c>
	static void apply_on_category(F&& f)
	{
		__apply_on_category<F, PROJECTS, SOURCES, USED_FORMATS, EDITED_FORMATS>(std::forward<F>(f));
	}

	/*! \brief Return a source description from the settings current group.
	 */
	template <Category category>
	typename list_type<category>::type get_list();

	void clear(Category category, QList<int> indexes = QList<int>());

	std::tuple<QString, QStringList> get_string_from_entry(QString const& string) const;
	std::tuple<QString, QStringList>
	get_string_from_entry(PVCore::PVSerializedSource const& sd) const;
	std::tuple<QString, QStringList> get_string_from_entry(PVRush::PVFormat const& f) const;

	void clear_missing_files();

  private:
	// Helper function to call a generic lambda on every category at compile time.
	// Could be improve with generic lambda (cxx14) for now, we use only functors ...

	template <class F, Category... c>
	static void __apply_on_category(F&& f)
	{
		int _[] __attribute__((unused)) = {(f.template call<c>(), 1)...};
	}

	PVCore::PVSerializedSource deserialize_source_description();

	/*! \brief Get the best source timestamp to replace (oldest, matching the same source
	 * description or 0).
	 *  \return The timestamp to replace or 0 if the list count is less than the maximum allowed
	 * items.
	 */
	uint64_t get_source_timestamp_to_replace(const PVRush::PVSourceDescription& source_description);

  private:
	/*! \brief Return a list of recent items of a given category as a list of QString.
	 */
	template <Category category>
	typename list_type<category>::type items_list() const;

	/**
	 * Remove value in recent file when pointed file is missing.
	 */
	void remove_missing_files(Category category);

	/*! \brief Return the recent sources description as a list
	 */
	// FIXME : This function is not const as it required group to list sources and Qt doesn't
	// provide this interface
	std::vector<PVCore::PVSerializedSource> sources_description_list();
	void remove_invalid_source();

  private:
	PVRecentItemsManager();
	PVRecentItemsManager(const PVRecentItemsManager&);
	PVRecentItemsManager& operator=(const PVRecentItemsManager&);

  public:
	std::array<sigc::signal<void()>, LAST> _add_item;

  private:
	QSettings _recents_settings;
	const int64_t _max_recent_items = 30;
	const QStringList _recents_items_keys = {"recent_projects", "recent_sources",
	                                         "recent_used_formats", "recent_edited_formats"};
};
} // namespace PVCore

#endif /* PVRECENTITEMSMANAGER_H_ */
