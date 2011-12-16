#ifndef PVCORE_PVSERIALIZEARCHIVE_H
#define PVCORE_PVSERIALIZEARCHIVE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVSerializeObject.h>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <vector>
#include <QVariant>
#include <QHash>

namespace PVCore {

class PVSerializeArchiveOptions;

class LibKernelDecl PVSerializeArchive: public boost::enable_shared_from_this<PVSerializeArchive>
{
	friend class PVSerializeObject;
public:
	enum archive_mode {
		read = 0,
		write
	};
	typedef uint32_t version_t;
public:
	PVSerializeArchive(version_t version);
	PVSerializeArchive(QString const& dir, archive_mode mode, version_t version);

	virtual ~PVSerializeArchive();

protected:
	PVSerializeArchive(const PVSerializeArchive& obj):
   		boost::enable_shared_from_this<PVSerializeArchive>(obj)
	{ assert(false); }

public:
	void open(QString const& dir, archive_mode mode);
	PVSerializeObject_p get_root();
	version_t get_version() const;
	void set_options(boost::shared_ptr<PVSerializeArchiveOptions> options) { _options = options; };
	void set_save_everything(bool save_everything) { _save_everything = save_everything; };
	// Finish function
	virtual void finish();

protected:
	bool is_writing() const { return _mode == write; }
	QString get_object_logical_path(PVSerializeObject const& so) { return so.get_logical_path(); };
	PVSerializeObject_p allocate_object(QString const& name, PVSerializeObject_p parent);
	bool must_write_object(PVSerializeObject const& parent, QString const& child);
	const PVSerializeArchiveOptions* get_options() const { return _options.get(); }
	QDir get_dir_for_object(PVSerializeObject const& so) const;
	PVSerializeObject_p get_object_by_path(QString const& path) const;

protected:
	// If you want to create another way of storing archives, you must reimplement these functions
	
	// Object create function
	virtual PVSerializeObject_p create_object(QString const& name, PVSerializeObject_p parent);
	// Attribute access functions
	virtual void attribute_write(PVSerializeObject const& so, QString const& name, QVariant const& obj);
	virtual void attribute_read(PVSerializeObject& so, QString const& name, QVariant& obj, QVariant const& def);
	virtual void list_attributes_write(PVSerializeObject const& so, QString const& name, std::vector<QVariant> const& obj);
	virtual void list_attributes_read(PVSerializeObject const& so, QString const& name, std::vector<QVariant>& obj);
	virtual void hash_arguments_write(PVSerializeObject const& so, QString const& name, PVArgumentList const& obj);
	virtual void hash_arguments_read(PVSerializeObject const& so, QString const& name, PVArgumentList& obj);
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
	boost::shared_ptr<PVSerializeArchiveOptions> _options;
	bool _save_everything;
};

}

#endif
