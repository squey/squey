#ifndef PVCORE_PVREGISTRABLE_CLASS
#define PVCORE_PVREGISTRABLE_CLASS

#include <pvkernel/core/general.h>
#include <boost/shared_ptr.hpp>

namespace PVCore {

// Forward declaration
template <class RegAs> class PVClassLibrary;

/*! \brief Define a class type that is "registreable". 
 *
 * \section background Background
 * What is done here is mainly the implementation of a factory pattern
 * \sa http://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Virtual_Constructor
 *
 * \section description Description
 * A registrable class is a class whose children can be
 * exported into plugins.
 *
 * For instance, here is an example of a definition of a plugin interface \c MyPluginInterface, whose
 * implementations will be defined in separate libraries :
 *
 * \code
 * class MyPluginInterface: public PVRegistrableClass<MyPluginInterface>
 * {
 * public:
 *	virtual void funcToImplement() = 0;
 * };
 * \endcode
 *
 *
 * Then, an implementation of \c MyPluginInterface would be:
 *
 * \code
 * class MyPluginImplementation: public MyPublicInterface
 * {
 * public:
 *	void funcToImplement() {
 *		// Do whatever the plugin is supposed to do.
 *	}
 *
 *	CLASS_REGISTRABLE(MyPluginInterface)
 * };
 * \endcode
 *
 * The CLASS_REGISTRABLE macro defines:
 * <ul>
 * <li>the \c p_type type, as a \c MyPluginInterface boost shared pointer.</li>
 * <li>the \c _clone_me method, that is used by the \ref clone method for polymorphic object cloning. The copy is made using the copy constructor of MyPluginImplementation.</li>
 * </ul>
 *
 * The CLASS_REGISTRABLE macro can be replaced by CLASS_REGISTRABLE_NOCOPY if no copy constructor is available for \c MyPluginImplementation.
 *
 * \section register Registration process
 * The registration process is handled by the \ref{PVClassLibrary} class.
 *
 * The REGISTER_CLASS macro must be used. For instance, here is how to register our previous MyPluginImplementation class:
 * \code
 * REGISTER_CLASS("unique-key-name", MyPluginImplementation);
 * \endcode
 *
 * This function should only be called once. The key used for the registration name must be unique across all MyPluginInterface's implementations.
 *
 * For more options (as registration with different parents), see \ref PVClassLibrary.
 *
 * \section register-id Registration ID
 * After a class has been registered, it is assigned a unique integer identifier accross all the classes with the same registered type (\c MyPluginInterface in our example).
 * This identifier can be used to compare the type of two objects, or to use 
 * \section stdmap-keys Use plugins as keys of an std::map object
 * \section tags Tags
 * Tags can be associated to registrable classes implementations.
 * For instance, to associate the tag "MyNewTag" with the description "This is a new tag" to \c MyPluginImplementation, the DECLARE_TAG macro is used:
 * \code
 * DECLARE_TAG("MyNewTag", "This is a new tag", MyPluginImplementation);
 * \endcode
 *
 * For more informations, see \ref PVTag.
 */
template <typename RegAs_>
class PVRegistrableClass
{
	template <class RegAs>
	friend class PVCore::PVClassLibrary;
public:
	typedef RegAs_ RegAs;
	typedef boost::shared_ptr< PVRegistrableClass<RegAs_> > p_type;
	typedef PVRegistrableClass<RegAs_> base_registrable;
	typedef int reg_id_t;
public:
	PVRegistrableClass()
	{
		__registered_class_id = -1;
	}

	/*! \brief Polymorphically clone this object.
	 * \tparam Tc the type of shared_pointer that will be returned. Tc must be at least a parent of this class (or this class itself).
	 * \return a shared pointer to a Tc object
	 * \sa _clone_me
	 */
	template <typename Tc>
	boost::shared_ptr<Tc> clone() const
	{
		base_registrable* rc = _clone_me();
		assert(rc);
		rc->__registered_class_name = __registered_class_name;
		Tc* ret = dynamic_cast<Tc*>(rc);
		assert(ret);
		return boost::shared_ptr<Tc>(ret);
	}
	inline QString const& registered_name() const { return __registered_class_name; }
	inline reg_id_t registered_id() const { return __registered_class_id; }
public:
	virtual bool operator<(const base_registrable& other) const
	{
		assert((registered_id() != -1) && (other.registered_id() != -1));
		return registered_id() < other.registered_id();
	}
	virtual bool operator==(const base_registrable& other) const
	{
		assert((registered_id() != -1) && (other.registered_id() != -1));
		return registered_id() == other.registered_id();
	}
protected:
	virtual base_registrable* _clone_me() const
	{
		// This is done so that this class can be instanciated and used as a key for std::map
		assert(false);
		return NULL;
	}
protected:
	QString __registered_class_name;
	// This `id' is set when the class is registered, and is unique accross the classes of the same registered type.
	reg_id_t __registered_class_id;
};

}

#define CLASS_REGISTRABLE(T) \
	public:\
		typedef boost::shared_ptr<T> p_type;\
	protected:\
		virtual T::base_registrable* _clone_me() const { T* ret = new T(*this); return ret; }\

#define CLASS_REGISTRABLE_NOCOPY(T) \
	public:\
		typedef boost::shared_ptr<T> p_type;\
	protected:\
		virtual T::base_registrable* _clone_me() const\
		{\
			T* ret = new T();\
			ret->__registered_class_name = __registered_class_name;\
			ret->__registered_class_id = __registered_class_id;\
			return ret;\
		}

#endif
