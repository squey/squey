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

#ifndef PVCORE_PVSERIALIZEARCHIVE_H
#define PVCORE_PVSERIALIZEARCHIVE_H

#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/core/PVSerializeObject.h>

#include <QVariant>
#include <QHash>
#include <QDir>
#include <QSettings>

#include <cassert>
#include <memory>
#include <vector>

namespace PVCore
{

class PVSerializeArchive
{
	friend class PVSerializeObject;

  public:
	enum archive_mode { read = 0, write };
	typedef uint32_t version_t;

  public:
	explicit PVSerializeArchive(version_t version, bool save_log_file = false);
	PVSerializeArchive(QString const& dir,
	                   archive_mode mode,
	                   version_t version,
	                   bool save_log_file);
	PVSerializeArchive(const PVSerializeArchive& obj) = delete;

	virtual ~PVSerializeArchive();

  public:
	void open(QString const& dir, archive_mode mode);
	PVSerializeObject_p get_root();
	version_t get_version() const;

	// Repairable errors
	void set_repaired_value(std::string const& path, std::string const& value)
	{
		_repaired[path] = value;
	}
	std::map<std::string, std::string> const& get_repaired_value() const { return _repaired; }
	void set_current_status(std::string const& s) { _current_status = s; }
	std::string const& get_current_status() const { return _current_status; }

  protected:
	bool is_writing() const { return _mode == write; }
	PVSerializeObject_p allocate_object(QString const& name, PVSerializeObject* parent);
	QDir get_dir_for_object(PVSerializeObject const& so) const;

  protected:
	// If you want to create another way of storing archives, you must reimplement these functions

	// Object create function
	virtual PVSerializeObject_p create_object(QString const& name, PVSerializeObject* parent);
	// Attribute access functions
	void attribute_write(PVSerializeObject const& so, QString const& name, QVariant const& obj);
	QVariant attribute_read(PVSerializeObject& so, QString const& name);

	virtual void list_attributes_write(PVSerializeObject const& so,
	                                   QString const& name,
	                                   std::vector<QVariant> const& obj);
	virtual void list_attributes_read(PVSerializeObject const& so,
	                                  QString const& name,
	                                  std::vector<QVariant>& obj);
	virtual void hash_arguments_write(PVSerializeObject const& so,
	                                  QString const& name,
	                                  PVArgumentList const& obj);
	void hash_arguments_read(PVSerializeObject const& so,
	                         QString const& name,
	                         PVArgumentList& obj,
	                         PVArgumentList const& def_args);
	size_t buffer_read(PVSerializeObject const& so, QString const& name, void* buf, size_t n);
	size_t
	buffer_write(PVSerializeObject const& so, QString const& name, void const* buf, size_t n);
	QString file_read(PVSerializeObject const& so, QString const& name);
	void file_write(PVSerializeObject const& so, QString const& name, QString const& path);

	bool save_log_file() const { return _save_log_files; }

  private:
	void init();
	void create_attributes(PVSerializeObject const& so);
	QString get_object_config_path(PVSerializeObject const& so) const;

  protected:
	PVSerializeObject_p _root_obj;
	archive_mode _mode;
	QString _root_dir;
	version_t _version;
	bool _is_opened;
	QHash<QString, QSettings*> _objs_attributes;

	/*! \brief Store a hash of object paths (as strings) to the real PVSerializeObject pointer
	 */
	QHash<QString, PVSerializeObject_p> _objects;

  private:
	std::map<std::string, std::string> _repaired; //!< Saved repaired value (path, value)

	bool _save_log_files;
	std::string _current_status; //!< Description about where we are in the serialization process.
};
} // namespace PVCore

#endif
