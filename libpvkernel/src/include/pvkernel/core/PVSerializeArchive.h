/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSERIALIZEARCHIVE_H
#define PVCORE_PVSERIALIZEARCHIVE_H

#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
#include <pvkernel/core/PVSerializeObject.h>

#include <QVariant>
#include <QHash>

#include <cassert>
#include <memory>
#include <vector>

namespace PVCore
{

class PVSerializeArchiveOptions;

class PVSerializeArchive
{
	friend class PVSerializeObject;

  public:
	enum archive_mode { read = 0, write };
	typedef uint32_t version_t;

  public:
	PVSerializeArchive(version_t version);
	PVSerializeArchive(QString const& dir, archive_mode mode, version_t version);
	PVSerializeArchive(const PVSerializeArchive& obj) = delete;

	virtual ~PVSerializeArchive();

  public:
	void open(QString const& dir, archive_mode mode);
	PVSerializeObject_p get_root();
	version_t get_version() const;
	void set_options(std::shared_ptr<PVSerializeArchiveOptions> options) { _options = options; };

	// Repairable errors
	void set_repaired_value(std::string const& path, std::string const& value)
	{
		_repaired = std::make_pair(path, value);
	}
	std::string const& get_repaired_path() const { return _repaired.first; }
	std::string const& get_repaired_value() const { return _repaired.second; }
	void set_current_status(std::string const& s) { _current_status = s; }
	std::string const& get_current_status() const { return _current_status; }

  protected:
	bool is_writing() const { return _mode == write; }
	QString get_object_logical_path(PVSerializeObject const& so) { return so.get_logical_path(); };
	PVSerializeObject_p allocate_object(QString const& name, PVSerializeObject* parent);
	bool must_write_object(PVSerializeObject const& parent, QString const& child);
	const PVSerializeArchiveOptions* get_options() const { return _options.get(); }
	QDir get_dir_for_object(PVSerializeObject const& so) const;

  protected:
	// If you want to create another way of storing archives, you must reimplement these functions

	// Object create function
	virtual PVSerializeObject_p create_object(QString const& name, PVSerializeObject* parent);
	// Attribute access functions
	virtual void
	attribute_write(PVSerializeObject const& so, QString const& name, QVariant const& obj);
	virtual void
	attribute_read(PVSerializeObject& so, QString const& name, QVariant& obj, QVariant const& def);
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
	virtual size_t buffer(PVSerializeObject const& so, QString const& name, void* buf, size_t n);
	virtual void file(PVSerializeObject const& so, QString const& name, QString& path);

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
	std::pair<std::string, std::string> _repaired; //!< Saved repaired value (path, value)

	std::shared_ptr<PVSerializeArchiveOptions> _options;
	std::string _current_status; //!< Description about where we are in the serialization process.
};
}

#endif
