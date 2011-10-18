#ifndef PVCORE_PVSERIALIZEOBJECT_H
#define PVCORE_PVSERIALIZEOBJECT_H

#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVSerializeArchiveExceptions.h>

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <QDir>
#include <QSettings>

#include <vector>

namespace PVCore {

class PVSerializeArchive;
typedef boost::shared_ptr<PVSerializeArchive> PVSerializeArchive_p;

class LibKernelDecl PVSerializeObjectFileError
{
public:
	PVSerializeObjectFileError(QFile const& file)
	{
		_msg = QString("Error with file '%1': %2 (%3)").arg(file.fileName()).arg(file.errorString()).arg(file.error());
	}

	QString const& what() const { return _msg; }
protected:
	QString _msg;
};

class LibKernelDecl PVSerializeObject: public boost::enable_shared_from_this<PVSerializeObject>
{
	friend class PVSerializeArchive;
public:
	typedef boost::shared_ptr<PVSerializeObject> p_type;
	typedef QHash<QString, p_type> list_childs_t;

protected:
	PVSerializeObject(QString const& logical_path, PVSerializeArchive_p parent_ar, p_type parent = p_type());

private:
	// Private copy-constructor, as these objects must always have been created by
	// PVSerializeArchive::create_object
	PVSerializeObject(const PVSerializeObject& src);

public:
	/*! \brief Split reading and writing serialization process
	 *  This can be called by any object being serialized to call different function for the writing and reading process.
	 */
	template <typename T>
	void split(T& obj)
	{
		if (is_writing()) {
			obj.serialize_write(*this);
		}
		else {
			obj.serialize_read(*this, get_version());
		}
	}

	bool is_writing() const;
	bool is_optional() const;
	bool must_write() const;
	void set_write(bool write);
	QString const& description() const;
	list_childs_t const& childs() const;
	const p_type get_child_by_name(QString const& name) const;
	QString const& get_logical_path() const;
	p_type parent();

public:
	/*! \brief Declare a new object to serialize that can be optionally saved, with a description.
	 *  \param[in] name Name of the object to serialize
	 *  \param[in][out] obj Object to load/save
	 *  \param[in] optional Can this object be optionally saved
	 *  \param[in] desc Description of the object (for usage in a GUI for instance)
	 *  \return If this object can be optionaly saved and an archive is being read, it returns true if the object exists in this archive and was successfully read.
	 *  This method will create a new PVSerializeObject and call the
	 *  serialize_(read/write) method of obj with this new PVSerializeObject.
	 */
	template <typename T>
	bool object(QString const& name, T& obj, QString const& desc = QString(), bool optional = false);

	/*! \brief Declare a list to serialize. T must be an STL-compliant container. T::value_type must be serializable.
	 *  \param[in] name Name of the list to serialize
	 *  This method declare a list of object to serialize.
	 *  This will be represented in this form:
	 *    list_name
	 *    |
	 *    -- 0
	 *       |
	 *       -- serialization of first object
	 *    -- 1
	 *       |
	 *       -- serialization of the second object
	 *    ...
	 */
	template <typename T>
	void list(QString const& name, T& obj);

	/*! \brief Declare a QHash to serialize. V must be serializable.
	 *  \param[in] name Name of the QHash to serialize.
	 *  This method declare a QHash of object to serialize.
	 *  K has to be convertible to a QVariant.
	 *  This will be represented in this form:
	 *    hash_name
	 *    |
	 *    -- key of first object
	 *       |
	 *       -- serialization of first object
	 *    -- key of second object
	 *       |
	 *       -- serialization of the second object
	 *    ...
	 */
	template <typename K, typename V>
	void hash(QString const& name, QHash<K,V>& obj);

	/*! \brief Declare an attribute to load/save in the 'configuration' of the object. T must be convertible to and from a QVariant
	 *  \param[in] name Name of the attribute
	 *  \param[in][out] obj Attribute to save
	 *  This attribute will be read/saved in the 'config.ini' file associated with the object.
	 */
	template <class T>
	void attribute(QString const& name, T& obj, T const& def = T());

	/*! \brief Declare a list of attributes to load/save in the 'configuration' of the object. T must be a STL-compliant container. T::value_type must be convertible to and from a QVariant.
	 *  \param[in] name Name of the list of attributes
	 *  \param[in][out] obj List of attributes to save
	 *  These attributes will be read/saved in the 'config.ini' file associated with the object.
	 */
	template <class T>
	void list_attributes(QString const& name, T& obj);

	/*! \brief Read/save a buffer for this object
	 *  \param[in] name Name of the buffer. This will be used for the underlying filename.
	 *  \param[in][out] buf Original/destination buffer
	 *  \param[in] n Size of the buffer
	 *  \return The number of bytes read/written
	 *  This will read/save the buffer pointed by buf. name is used as the underlying filename.
	 */
	size_t buffer(QString const& name, void* buf, size_t n);

	/*! \brief Include an existing file, given its path.
	 *  \param[in] name Name of this file. This will be used as the underlying destination filename.
	 *  \param[in][out] path Path to the file. When reading the archive, this is set to the extracted file path.
	 */
	void file(QString const& name, QString& path);

private:
	p_type create_object(QString const& name, bool optional = false, QString const& desc = QString());
	uint32_t get_version() const;
	void attribute_write(QString const& name, QVariant const& obj);
	void attribute_read(QString const& name, QVariant& obj, QVariant const& def);
	void list_attributes_write(QString const& name, std::vector<QVariant> const& list);
	void list_attributes_read(QString const& name, std::vector<QVariant>& list);
	bool must_write_child(QString const& name);

	template <typename T>
	void call_serialize(T& obj, p_type new_obj) { obj.serialize(*new_obj, get_version()); }

	template <typename T>
	void call_serialize(boost::shared_ptr<T>& obj, p_type new_obj)
	{
		if (!obj) {
			assert(!is_writing());
			obj.reset(new T());
		}
		obj->serialize(*new_obj, get_version());
	}

private:
	/*! \brief Parent archive pointer
	 */
	PVSerializeArchive_p _parent_ar;

	/*! \brief Parent serialization object
	 */
	p_type _parent;

	/*! \brief Logicial path within the archive
	 *  This path is set by PVSerializeArchive::create_object and is independent of the
	 *  final root directory and any operating system rules.
	 *  This can be used to uniquely identify an object.
	 */
	QString _logical_path;

	/*! \brief Specifies whether this object is optional or not. Set by create_object.
	 */
	bool _is_optional;
	
	/*! \brief Specifies a description of this object.
	 */
	QString _desc;

	/*! \brief If this object is optional, specifies whether this object must be written or not.
	 */
	bool _must_write;

	/*! \brief Hash of the childs of this object. The key is their name.
	 */
	list_childs_t _childs;
};

typedef PVSerializeObject::p_type PVSerializeObject_p;

template <typename T>
bool PVSerializeObject::object(QString const& name, T& obj, QString const& desc, bool optional)
{
	if (optional && is_writing()) {
		if (!must_write_child(name)) {
			return false;
		}
	}
	QString desc_ = (desc.isNull()) ? name:desc;
	p_type new_obj;
	try {
		new_obj = create_object(name, optional, desc_);
	}
	catch (PVSerializeArchiveErrorNoObject &e) {
		if (!optional && !is_writing()) {
			throw e;
		}
		return false;
	}
	call_serialize(obj, new_obj);
	return true;
}

template <typename T>
void PVSerializeObject::list(QString const& name, T& obj)
{
	PVSerializeObject_p list_obj = create_object(name);
	if (is_writing()) {
		typename T::iterator it;
		int idx = 0;
		for (it = obj.begin(); it != obj.end(); it++) {
			typename T::value_type& v = *it;
			PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
			call_serialize(v, new_obj);
			idx++;
		}
	}
	else {
		obj.clear();
		int idx = 0;
		try {
			while (true) {
				typename T::value_type v;
				PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
				call_serialize(v, new_obj);
				obj.push_back(v);
				idx++;
			}
		}
		catch (PVSerializeArchiveError& e) {
			return;
		}
	}
}

template <typename K, typename V>
void PVSerializeObject::hash(QString const& name, QHash<K,V>& obj)
{
	if (is_writing()) {
		typename QHash<K,V>::const_iterator it;
		for (it = obj.begin(); it != obj.end(); it++) {
			PVSerializeObject_p new_obj = create_object(it.key());
			it.value().serialize(*new_obj);
		}
	}
	else {
	}
}

template <class T>
void PVSerializeObject::attribute(QString const& name, T& obj, T const& def)
{
	if (is_writing()) {
		attribute_write(name, QVariant(obj));
	}
	else {
		QVariant v;
		attribute_read(name, v, QVariant(def));
		obj = v.value<T>();
	}
}

template <class T>
void PVSerializeObject::list_attributes(QString const& name, T& obj)
{
	if (is_writing()) {
		std::vector<QVariant> list;
		list.reserve(obj.size());
		typename T::const_iterator it;
		for (it = obj.begin(); it != obj.end(); it++) {
			typename T::value_type const& v = *it;
			list.push_back(QVariant(v));
		}
		list_attributes_write(name, list);
	}
	else {
		std::vector<QVariant> list;
		list_attributes_read(name, list);
		std::vector<QVariant>::iterator it;
		for (it = list.begin(); it != list.end(); it++) {
			obj.push_back(it->value<typename T::value_type>());
		}
	}
}

}

// Conveniance macros
#define PVSERIALIZEOBJECT_SPLIT\
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)\
	{\
		so.split(*this);\
	}\

#endif
