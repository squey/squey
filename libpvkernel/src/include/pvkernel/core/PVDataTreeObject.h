/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDATATREEOBJECT_H_
#define PVDATATREEOBJECT_H_

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVEnableSharedFromThis.h>

#include <algorithm>
#include <list>

namespace PVCore
{

namespace __impl
{
template <class T, class B>
struct ChildrenAccessor {
	static size_t size(std::list<PVSharedPtr<T>> const& c)
	{
		return std::accumulate(c.begin(), c.end(), 0UL, [](size_t cum, PVSharedPtr<T> const& c1) {
			using child_t = typename std::remove_reference<
			    typename std::remove_cv<decltype(*c1->get_children().begin()->get())>::type>::type;
			return cum + ChildrenAccessor<child_t, B>::size(c1->template get_children());
		});
	}
};

template <class T>
struct ChildrenAccessor<T, T> {
	static size_t size(std::list<PVSharedPtr<T>> const& c) { return c.size(); }
};
}

template <class Child, class Derived>
class PVDataTreeParent
{
  public:
	template <class... T>
	PVSharedPtr<Child> emplace_add_child(T&&... t)
	{
		_children.push_back(PVSharedPtr<Child>(new Child(static_cast<Derived*>(this), t...)));
		return _children.back();
	}

	std::list<PVSharedPtr<const Child>> get_children() const
	{
		return {_children.begin(), _children.end()};
	}

	std::list<PVSharedPtr<Child>>& get_children() { return _children; }

	void remove_child(Child const& child)
	{
		std::remove(_children.begin(), _children.end(), child.shared_from_this());
	}

	void remove_all_children() { _children.clear(); }

	template <class T = Child>
	size_t size() const
	{
		return __impl::ChildrenAccessor<Child, T>::size(_children);
	}

  private:
	std::list<PVSharedPtr<Child>> _children;
};

namespace __impl
{
template <class T, class B>
struct ParentAccessor {
	static T const* access(B const* c) { return c->template get_parent<T>(); }
	static T* access(B* c) { return c->template get_parent<T>(); }
};

template <class T>
struct ParentAccessor<T, T> {
	static T const* access(T const* c) { return c; }
	static T* access(T* c) { return c; }
};
}

template <class Parent, class Derived>
class PVDataTreeChild
{
  public:
	PVDataTreeChild(Parent* parent) : _parent(parent) {}

	void remove_from_tree()
	{
		Derived* me = static_cast<Derived*>(this);
		_parent->remove_child(*me);
	}

	template <class T = Parent>
	T const* get_parent() const
	{
		return __impl::ParentAccessor<T, Parent>::access(_parent);
	}

	template <class T = Parent>
	T* get_parent()
	{
		return __impl::ParentAccessor<T, Parent>::access(_parent);
	}

  private:
	Parent* _parent;
};
}

#endif /* PVDATATREEOBJECT_H_ */
