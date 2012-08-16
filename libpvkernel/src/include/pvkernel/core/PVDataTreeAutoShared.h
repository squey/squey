#ifndef PVCORE_PVDATATREEAUTOSHARED_H
#define PVCORE_PVDATATREEAUTOSHARED_H

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVTypeTraits.h>

namespace PVCore {

template <class T>
class PVDataTreeAutoShared: public PVCore::PVSharedPtr<T>
{
public:
	template <typename U = T, typename std::enable_if<U::has_parent == true, int>::type = 0, typename... Tparams>
	PVDataTreeAutoShared(typename U::pparent_t const& parent, Tparams && ... params):
		PVCore::PVSharedPtr<T>(new T(std::forward<Tparams>(params)...))
	{
		this->get()->set_parent(parent);
	}

	template <typename U = T, typename std::enable_if<U::has_parent == false, int>::type = 0, typename... Tparams>
	PVDataTreeAutoShared(Tparams && ... params):
		PVCore::PVSharedPtr<U>(new U(std::forward<Tparams>(params)...))
	{ }

	PVDataTreeAutoShared():
		PVCore::PVSharedPtr<T>(new T())
	{ }


public:
	PVDataTreeAutoShared(PVCore::PVSharedPtr<T> const& o):
		PVCore::PVSharedPtr<T>(o)
	{ }

public:
	static PVDataTreeAutoShared invalid() { return PVCore::PVSharedPtr<T>(); }

public:
	// Implicit conversion to PVCore::PVSharedPtr<T>
	inline operator PVCore::PVSharedPtr<T>&() { return *this; }
	inline operator PVCore::PVSharedPtr<T> const&() const { return *this; }
	inline PVDataTreeAutoShared& operator=(PVDataTreeAutoShared const& o) { PVCore::PVSharedPtr<T>::operator=(static_cast<PVCore::PVSharedPtr<T> const&>(o)); return *this; }
	inline PVDataTreeAutoShared& operator=(PVCore::PVSharedPtr<T> const& o) { PVCore::PVSharedPtr<T>::operator=(o); return *this; }
};

namespace PVTypeTraits {

template <class T>
struct remove_shared_ptr<PVDataTreeAutoShared<T> >
{
	typedef T type;
};

template <class T>
struct pointer<PVDataTreeAutoShared<T> >
{
	typedef PVDataTreeAutoShared<T> type;
	static inline type get(type obj) { return obj; }
};

template <class T>
struct pointer< PVDataTreeAutoShared<T>& >
{
	typedef  PVDataTreeAutoShared<T>& type;
	static inline type get(type obj) { return obj; }
};

template <class Y, class T>
struct dynamic_pointer_cast< PVDataTreeAutoShared<Y>,  PVDataTreeAutoShared<T> >
{
	typedef PVDataTreeAutoShared<T> org_pointer;
	typedef PVDataTreeAutoShared<Y> result_pointer;
	static result_pointer cast(org_pointer const& p) { return result_pointer(PVCore::dynamic_pointer_cast<Y>(p)); }
};

}

}

#endif
