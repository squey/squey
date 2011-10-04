#ifndef PVCORE_PVSERIALIZEARCHIVE_H
#define PVCORE_PVSERIALIZEARCHIVE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVSerializeObject.h>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

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

public:
	void open(QString const& dir, archive_mode mode);
	PVSerializeObject_p get_root();
	version_t get_version();
	virtual void finish();

protected:
	PVSerializeObject_p create_object(QString const& name, PVSerializeObject_p parent);
	bool is_writing() { return _mode == write; }

private:
	void init();

protected:
	PVSerializeObject_p _root_obj;
	archive_mode _mode;
	QString _root_dir;
	version_t _version;
	bool _is_opened;
};

}

#endif
