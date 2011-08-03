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
public:
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
	QString const& registered_name() { return __registered_class_name; }
protected:
	virtual base_registrable* _clone_me() const = 0;
protected:
	QString __registered_class_name;
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
		virtual T::base_registrable* _clone_me() const { T* ret = new T(); return ret; }\

#endif
