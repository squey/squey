#ifndef PVCORE_PVREGISTRABLE_CLASS
#define PVCORE_PVREGISTRABLE_CLASS

#include <pvkernel/core/general.h>
#include <boost/shared_ptr.hpp>

namespace PVCore {

// Forward declaration
template <class RegAs> class PVClassLibrary;

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
