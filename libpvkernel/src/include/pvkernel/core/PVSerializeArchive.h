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

private:
	PVSerializeArchive(const PVSerializeArchive& obj):
   		boost::enable_shared_from_this<PVSerializeArchive>(obj)
	{ assert(false); }

public:
	void open(QString const& dir, archive_mode mode);
	PVSerializeObject_p get_root();
	version_t get_version() const;
	virtual void finish();

protected:
	PVSerializeObject_p create_object(QString const& name, PVSerializeObject_p parent);
	bool is_writing() const { return _mode == write; }

protected:
	// Attribute access functions
	virtual void attribute_write(PVSerializeObject const& so, QString const& name, QVariant const& obj);
	virtual void attribute_read(PVSerializeObject& so, QString const& name, QVariant& obj, QVariant const& def);
	virtual void list_attributes_write(PVSerializeObject const& so, QString const& name, std::vector<QVariant> const& obj);
	virtual void list_attributes_read(PVSerializeObject const& so, QString const& name, std::vector<QVariant>& obj);

private:
	void init();
	void create_attributes(PVSerializeObject const& so);

protected:
	PVSerializeObject_p _root_obj;
	archive_mode _mode;
	QString _root_dir;
	version_t _version;
	bool _is_opened;
	QHash<QString, QSettings*> _objs_attributes;
};

}

#endif
