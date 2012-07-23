/**
 * \file PVSerializeArchive.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVSERIALIZEARCHIVE_H
#define PVCORE_PVSERIALIZEARCHIVE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchiveExceptions.h>
#include <pvkernel/core/PVSerializeArchiveFixError.h>
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
	typedef QList<boost::shared_ptr<PVSerializeArchiveFixError> > list_errors_t;
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

	// Repairable errors
	inline bool has_repairable_errors() const { return _repairable_errors.size() > 0; }
	template <class T>
	bool has_repairable_errors_of_type() const;
	inline list_errors_t const& get_repairable_errors() const { return _repairable_errors; }
	template <class T>
	list_errors_t get_repairable_errors_of_type() const;

protected:
	bool is_writing() const { return _mode == write; }
	QString get_object_logical_path(PVSerializeObject const& so) { return so.get_logical_path(); };
	PVSerializeObject_p allocate_object(QString const& name, PVSerializeObject* parent);
	bool must_write_object(PVSerializeObject const& parent, QString const& child);
	const PVSerializeArchiveOptions* get_options() const { return _options.get(); }
	QDir get_dir_for_object(PVSerializeObject const& so) const;
	PVSerializeObject_p get_object_by_path(QString const& path) const;
	bool object_exists_by_path(QString const& path) const;

protected:
	// If you want to create another way of storing archives, you must reimplement these functions
	
	// Object create function
	virtual PVSerializeObject_p create_object(QString const& name, PVSerializeObject* parent);
	// Attribute access functions
	virtual void attribute_write(PVSerializeObject const& so, QString const& name, QVariant const& obj);
	virtual void attribute_read(PVSerializeObject& so, QString const& name, QVariant& obj, QVariant const& def);
	virtual void list_attributes_write(PVSerializeObject const& so, QString const& name, std::vector<QVariant> const& obj);
	virtual void list_attributes_read(PVSerializeObject const& so, QString const& name, std::vector<QVariant>& obj);
	virtual void hash_arguments_write(PVSerializeObject const& so, QString const& name, PVArgumentList const& obj);
	virtual void hash_arguments_read(PVSerializeObject const& so, QString const& name, PVArgumentList& obj, PVArgumentList const& def_args);
	virtual size_t buffer(PVSerializeObject const& so, QString const& name, void* buf, size_t n);
	virtual void buffer_path(PVSerializeObject const& so, QString const& name, QString& path);
	virtual void file(PVSerializeObject const& so, QString const& name, QString& path);

	// Called by PVSerializeObject
	void repairable_error(boost::shared_ptr<PVSerializeArchiveFixError> const& error);
	void error_fixed(PVSerializeArchiveFixError* error);

	QString get_object_path_in_archive(const void* obj_ptr) const;

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

	/*! \brief List of the declared repairable errors.
	 *  \sa repairable_error
	 */
	list_errors_t _repairable_errors;

};

template <class T>
bool PVSerializeArchive::has_repairable_errors_of_type() const
{
	typedef typename PVTypeTraits::pointer<T>::type Tp;
	list_errors_t::const_iterator it;
	for (it = _repairable_errors.begin(); it != _repairable_errors.end(); it++) {
		if ((*it)->exception_of_type<T>()) {
			return true;
		}
	}
	return false;
}

template <class T>
PVSerializeArchive::list_errors_t PVSerializeArchive::get_repairable_errors_of_type() const
{
	typedef typename PVTypeTraits::pointer<T>::type Tp;
	list_errors_t ret;
	list_errors_t::const_iterator it;
	for (it = _repairable_errors.begin(); it != _repairable_errors.end(); it++) {
		if ((*it)->exception_of_type<T>()) {
			ret.push_back(*it);
		}
	}
	return ret;
}

}

#endif
