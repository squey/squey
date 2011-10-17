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

protected:
	PVSerializeObject(QDir const& path, PVSerializeArchive_p parent_ar, p_type parent = p_type());
	QString get_config_path() const;

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

public:
	/*! \brief Declare a new object to serialize
	 *  \param[in] name Name of the object to serialize
	 *  \param[in][out] obj Object to load/save
	 *  This method will create a new PVSerializeObject and call the
	 *  serialize_(read/write) method of obj with this new PVSerializeObject.
	 */
	template <typename T>
	void object(QString const& name, T& obj);

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

private:
	p_type create_object(QString const& name);
	uint32_t get_version() const;
	void attribute_write(QString const& name, QVariant const& obj);
	void attribute_read(QString const& name, QVariant& obj, QVariant const& def);
	void list_attributes_write(QString const& name, std::vector<QVariant> const& list);
	void list_attributes_read(QString const& name, std::vector<QVariant>& list);

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

protected:
	QDir const& get_path() const { return _path; }

private:
	/*! \brief Parent archive pointer
	 */
	PVSerializeArchive_p _parent_ar;

	/*! \brief Parent serialization object
	 */
	p_type _parent;

	/*! \brief Full path within the archive.
	 *  This path is set by PVSerializeArchive::create_object according to
	 *  the parent object.
	 */
	QDir _path;
};

typedef PVSerializeObject::p_type PVSerializeObject_p;

template <typename T>
void PVSerializeObject::object(QString const& name, T& obj)
{
	p_type new_obj = create_object(name);
	call_serialize(obj, new_obj);
}

template <typename T>
void PVSerializeObject::list(QString const& name, T& obj)
{
	PVSerializeObject_p list_obj = create_object(name);
	if (is_writing()) {
		typename T::iterator it;
		int idx = 0;
		QDir dir_obj(_path);
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
		QDir dir_obj(_path);
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
