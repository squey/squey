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

#ifndef __PVCORE_PVNRAWCACHEMANAGER_H__
#define __PVCORE_PVNRAWCACHEMANAGER_H__

#include <memory>
#include <functional>

#include <QDirIterator>
#include <QSettings>

class QString;
#include <QStringList>

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
	QStringList visit_nraw_folders(const QString& base_directory,
	                               const QString& name_filter,
	                               std::function<bool(QDirIterator& it)> f);
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
} // namespace PVRush

#endif // __PVCORE_PVNRAWCACHEMANAGER_H__
