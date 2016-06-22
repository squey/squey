/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDATATREEOBJECT_H_
#define PVDATATREEOBJECT_H_

#include <algorithm>
#include <list>

namespace PVCore
{

/**
 * it is a diamon base class for every child use to dynamic cast to correct type asap.
 */
class PVDataTreeObject
{
  public:
	/**
	 * Human readable description of a node.
	 */
	virtual std::string get_serialize_description() const = 0;
};

namespace __impl
{

/**
 * Helper class to have partially specialized methods when performing operations on child nodes.
 *
 * T is the current type of the list of nodes processed
 * B the required type node to handle.
 */
template <class T, class B>
struct ChildrenAccessor {
	/**
	 *  Accumulate size of sub-nodes
	 */
	static size_t size(std::list<const T*> const& c)
	{
		return std::accumulate(c.begin(), c.end(), 0UL, [](size_t cum, T const* c1) {
			using child_t = typename std::remove_cv<
			    typename std::remove_reference<decltype(**c1->get_children().begin())>::type>::type;
			return cum + ChildrenAccessor<child_t, B>::size(c1->template get_children());
		});
	}

	/**
	 * Accumulate nodes from sub-nodes
	 */
	static std::list<B*> children(std::list<T*>&& c)
	{
		std::list<B*> res;
		for (auto ch : c) {
			using child_t =
			    typename std::remove_reference<decltype(**ch->get_children().begin())>::type;
			res.splice(res.end(), ChildrenAccessor<child_t, B>::children(ch->get_children()));
		}
		return res;
	}
};

/**
 * Last case when required node is the same as current node
 */
template <class T>
struct ChildrenAccessor<T, T> {
	/**
	 * Size is the size of the children list
	 */
	static size_t size(std::list<const T*> const& c) { return c.size(); }

	/**
	 * Children are the ones in the current list.
	 */
	static std::list<T*> children(std::list<T*>&& c) { return c; }
};
}

/**
 * DataTree node as parent (containing children)
 */
template <class Child, class Derived>
class PVDataTreeParent : virtual public PVDataTreeObject
{
  public:
	PVDataTreeParent() = default;
	// No copy/move as child reparting would be required.
	PVDataTreeParent(PVDataTreeParent const&) = delete;
	PVDataTreeParent(PVDataTreeParent&&) = delete;
	PVDataTreeParent& operator=(PVDataTreeParent const&) = delete;
	PVDataTreeParent& operator=(PVDataTreeParent&&) = delete;

  public:
	template <class... T>
	Child& emplace_add_child(T&&... t)
	{
		_children.emplace_back(static_cast<Derived&>(*this), std::forward<T>(t)...);
		return _children.back();
	}

	template <class T = Child>
	std::list<const T*> get_children() const
	{
		std::list<const Child*> tmp_list;
		std::transform(_children.begin(), _children.end(), back_inserter(tmp_list),
		               [](Child const& p) { return &p; });
		return __impl::ChildrenAccessor<const Child, const T>::children(std::move(tmp_list));
	}

	template <class T = Child>
	std::list<T*> get_children()
	{
		std::list<Child*> tmp_list;
		std::transform(_children.begin(), _children.end(), back_inserter(tmp_list),
		               [](Child& p) { return &p; });
		return __impl::ChildrenAccessor<Child, T>::children(std::move(tmp_list));
	}

	void remove_child(Child& child)
	{
		_children.remove_if([&](Child const& c) { return &c == &child; });
	}

	void remove_all_children() { _children.clear(); }

	template <class T = Child>
	size_t size() const
	{
		return __impl::ChildrenAccessor<Child, T>::size(get_children());
	}

  private:
	std::list<Child> _children;
};

namespace __impl
{
template <class T, class B>
struct ParentAccessor {
	static T const& access(B const& c) { return c.template get_parent<T>(); }
	static T& access(B& c) { return c.template get_parent<T>(); }
};

template <class T>
struct ParentAccessor<T, T> {
	static T const& access(T const& c) { return c; }
	static T& access(T& c) { return c; }
};
}

template <class Parent, class Derived>
class PVDataTreeChild : virtual public PVDataTreeObject
{
  public:
	PVDataTreeChild(Parent& parent) : _parent(parent) {}
	// No copy/move as it should also be added to parent
	PVDataTreeChild(PVDataTreeChild const&) = delete;
	PVDataTreeChild(PVDataTreeChild&&) = delete;
	PVDataTreeChild& operator=(PVDataTreeChild const&) = delete;
	PVDataTreeChild& operator=(PVDataTreeChild&&) = delete;

  public:
	void remove_from_tree()
	{
		Derived* me = static_cast<Derived*>(this);
		_parent.remove_child(*me);
	}

	template <class T = Parent>
	T const& get_parent() const
	{
		return __impl::ParentAccessor<T, Parent>::access(_parent);
	}

	template <class T = Parent>
	T& get_parent()
	{
		return __impl::ParentAccessor<T, Parent>::access(_parent);
	}

  private:
	Parent& _parent;
};
}

#endif /* PVDATATREEOBJECT_H_ */
