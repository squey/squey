/**
 * \file datatree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>

#include <QList>

/******************************************************************************
 *
 * Tree classes declaration (A -> B -> C -> D)
 *
 *****************************************************************************/
// forward declarations
class A;
class B;
class C;
class D;

#define PVSERIALIZEOBJECT_SPLIT\
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)\
	{\
		so.split(*this);\
	}\

typedef typename PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<A>, B> data_tree_a_t;
class A : public data_tree_a_t
{
	friend class PVCore::PVDataTreeAutoShared<A>;

protected:
	A(int i = 0):
		data_tree_a_t(),
		_i(i)
	{}

public:
	virtual ~A() { std::cout << "~A(" << this << ")" << std::endl; }

	void save_to_file(QString const& path, bool save_everything = true)
	{
		PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::write, PICVIZ_ARCHIVES_VERSION));
		ar->set_save_everything(save_everything);
		QString name = "root";
		ar->get_root()->object(name, *this);
		ar->finish();
	}

	void load_from_file(QString const& path)
	{
		PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
		QString name = "root";
		ar->get_root()->object(name, *this);
	}

	virtual void serialize_write(PVCore::PVSerializeObject& so)
	{
		data_tree_a_t::serialize_write(so);

		so.attribute("_i", _i);
	}

	virtual void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
	{
		data_tree_a_t::serialize_read(so, v);

		so.attribute("_i", _i);
	}

	PVSERIALIZEOBJECT_SPLIT

	int get_i() { return _i; }

protected:
	void child_added(B& b);

private:
	int _i;
};

typedef typename PVCore::PVDataTreeObject<A, C> data_tree_b_t;
class B : public data_tree_b_t
{
	friend class PVCore::PVDataTreeAutoShared<B>;
	friend class A;

protected:
	B(int i = 0):
		data_tree_b_t(),
		_a_was_here(false),
		_i(i)
   	{}

public:
	virtual ~B() { std::cout << "~B(" << this << ")" << std::endl; }

	void set_parent_from_ptr(A* parent)
	{
		data_tree_b_t::set_parent_from_ptr(parent);
		_j = get_parent()->get_i()*2;
	}

	virtual void serialize_write(PVCore::PVSerializeObject& so)
	{
		data_tree_b_t::serialize_write(so);

		so.attribute("_i", _i);
		so.attribute("_j", _j);
	}

	virtual void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
	{
		data_tree_b_t::serialize_read(so, v);

		so.attribute("_i", _i);
		so.attribute("_j", _j);
	}

	PVSERIALIZEOBJECT_SPLIT

	int get_i() { return _i; }
	int get_j() { return _j; }
	void set_j(int j) { _j = j; }

	inline bool a_was_here() const { return _a_was_here; }

public:
	void f() const { std::cout << "B i = " << _i << std::endl; }
	inline int i() const { return _i; }

protected:
	bool _a_was_here;

private:
	int _i;
	int _j;
};

void A::child_added(B& b)
{
	b._a_was_here = true;
}


typedef typename PVCore::PVDataTreeObject<B, D> data_tree_c_t;
class C : public data_tree_c_t
{
	friend class PVCore::PVDataTreeAutoShared<C>;

protected:
	C(int i = 0):
		data_tree_c_t(),
		_i(i)
	{ }

public:
	virtual ~C() { std::cout << "~C(" << this << ")" << std::endl; }

	virtual void set_parent_from_ptr(B* parent)
	{
		data_tree_c_t::set_parent_from_ptr(parent);
		_j = get_parent()->get_i()*2;
	}

	virtual void serialize_write(PVCore::PVSerializeObject& so)
	{
		data_tree_c_t::serialize_write(so);

		so.attribute("_i", _i);
		so.attribute("_j", _j);
	}

	virtual void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
	{
		data_tree_c_t::serialize_read(so, v);

		so.attribute("_i", _i);
		so.attribute("_j", _j);
	}

	PVSERIALIZEOBJECT_SPLIT

	int get_i() { return _i; }
	int get_j() { return _j; }
	void set_j(int j) { _j = j; }

private:
	int _i;
	int _j;
};

typedef typename PVCore::PVDataTreeObject<C, PVCore::PVDataTreeNoChildren<D>> data_tree_d_t;
class D : public data_tree_d_t
{
	friend class PVCore::PVDataTreeAutoShared<D>;

protected:
	D(int i = 0):
		data_tree_d_t(),
		_i(i)
   	{ }

public:
	virtual ~D()
	{
		std::cout << "~D(" << this << ")" << std::endl;
	}

	virtual void set_parent_from_ptr(C* parent)
	{
		data_tree_d_t::set_parent_from_ptr(parent);
		_j = get_parent()->get_i()*2;
	}

	virtual void serialize_write(PVCore::PVSerializeObject& so)
	{
		data_tree_d_t::serialize_write(so);

		so.attribute("_i", _i);
		so.attribute("_j", _j);
	}

	virtual void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
	{
		data_tree_d_t::serialize_read(so, v);

		so.attribute("_i", _i);
		so.attribute("_j", _j);
	}

	PVSERIALIZEOBJECT_SPLIT

	int get_i() { return _i; }
	int get_j() { return _j; }
	void set_j(int j) { _j = j; }

private:
	int _i;
	int _j;
};


typedef typename A::p_type A_p;
typedef typename B::p_type B_p;
typedef typename C::p_type C_p;
typedef typename D::p_type D_p;

bool my_assert(bool res)
{
	assert(res);
	return res;
}

bool standard_use_case()
{
	//////////////////////////////////////////
	//  Test1 - Parent access
	//////////////////////////////////////////
	PVLOG_INFO("Constructing initial tree \n");
	A_p a1(4);
	A_p a2;
	B_p b1(a1, 5);
	B_p b2(a1);
	B_p b3(a2);
	C_p c(b1);
	D_p d(c);

	// Check "child_added"
	my_assert(b1->a_was_here());
	my_assert(b2->a_was_here());
	my_assert(b3->a_was_here());

	// Auto shared check
	B_p b1_other = b1;
	std::cout << "B1 other: " << b1_other.get() << " , b1: " << b1.get() << std::endl;
	my_assert(b1_other.get() == b1.get());
	my_assert(b1_other->i() == 5);
	b1_other = b2;
	std::cout << "B2 other: " << b1_other.get() << " , b2: " << b2.get() << std::endl;
	my_assert(b1_other.get() == b2.get());
	my_assert(b1_other->i() == 0);


	// Check invalid pointers
	C_p c_inv = C_p::invalid();
	my_assert(c_inv.get() == NULL);

	b1->f();

	std::cout << "a1=" << a1.get() << std::endl;
	std::cout << "a2=" << a2.get() << std::endl;
	std::cout << "b1=" << b1.get() << std::endl;
	std::cout << "b2=" << b2.get() << std::endl;
	std::cout << "b3=" << b3.get() << std::endl;
	std::cout << "c=" << c.get() << std::endl;
	std::cout << "d=" << d.get() << std::endl;

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump();
	std::cout << std::endl;

	bool parent_access = true;
	parent_access &= my_assert(b1->get_parent() == a1.get());
	parent_access &= my_assert(b2->get_parent() == a1.get());
	parent_access &= my_assert(b3->get_parent() == a2.get());
	parent_access &= my_assert(c->get_parent() == b1.get());
	parent_access &= my_assert(c->get_parent<A>() == a1.get());
	parent_access &= my_assert(d->get_parent() == c.get());
	parent_access &= my_assert(d->get_parent<B>() == b1.get());
	parent_access &= my_assert(d->get_parent<A>() == a1.get());
	parent_access &= my_assert(d->get_parent<C>() == c.get());
	PVLOG_INFO("Parent access passed: %d\n", parent_access);


	//////////////////////////////////////////
	//  Test2 - Children access
	//////////////////////////////////////////
	bool children_access = true;
	{
	auto a1_children = a1->get_children();
	children_access &= my_assert(a1_children.size() == 2 && a1_children[0] == b1 && a1_children[1] == b2);
	auto a2_children = a2->get_children();
	children_access &= my_assert(a2_children.size() == 1 && a2_children[0] == b3);
	auto b1_children = b1->get_children();
	children_access &= my_assert(b1_children.size() == 1 && b1_children[0] == c);
	children_access &= my_assert(b2->get_children().size() == 0);
	children_access &= my_assert(b3->get_children().size() == 0);
	auto c_children = c->get_children();
	children_access &= my_assert(c_children.size() == 1 && c_children[0] == d);
	}
	PVLOG_INFO("Children access passed: %d\n", children_access);


	//////////////////////////////////////////
	//  Test3 - Same parent
	//////////////////////////////////////////
	bool same_parent = true;
	//a1->set_parent(nullptr);
	//a2->set_parent(nullptr);
	b1->set_parent(a1);
	b1->set_parent(a1);
	b1->set_parent(a1);
	b3->set_parent(a2);
	//c->set_parent(b0);
	c->set_parent(b1);
	d->set_parent(c);

	{
	//same_parent &= my_assert(a1->get_parent() == nullptr);
	//same_parent &= my_assert(a2->get_parent() == nullptr);
	same_parent &= my_assert(b1->get_parent() == a1.get());
	same_parent &= my_assert(b2->get_parent() == a1.get());
	same_parent &= my_assert(b3->get_parent() == a2.get());
	same_parent &= my_assert(c->get_parent() == b1.get());
	same_parent &= my_assert(c->get_parent<A>() == a1.get());
	same_parent &= my_assert(d->get_parent() == c.get());
	same_parent &= my_assert(d->get_parent<B>() == b1.get());
	same_parent &= my_assert(d->get_parent<A>() == a1.get());
	same_parent &= my_assert(d->get_parent<C>() == c.get());
	auto a1_children = a1->get_children();
	same_parent &= my_assert(a1_children.size() == 2 && a1_children[0] == b1 && a1_children[1] == b2);
	auto a2_children = a2->get_children();
	same_parent &= my_assert(a2_children.size() == 1 && a2_children[0] == b3);
	auto b1_children = b1->get_children();
	same_parent &= my_assert(b1_children.size() == 1 && b1_children[0] == c);
	same_parent &= my_assert(b2->get_children().size() == 0);
	same_parent &= my_assert(b3->get_children().size() == 0);
	auto c_children = c->get_children();
	same_parent &= my_assert(c_children.size() == 1 && c_children[0] == d);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;


	PVLOG_INFO("Reaffecting same parent passed: %d\n", same_parent);

	//////////////////////////////////////////
	//  Test4 - Changing parent
	//////////////////////////////////////////
	bool changing_parent = true;
	b1->set_parent(a2);

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;


	{
	// a1 <-> b2
	auto a1_children = a1->get_children();
	changing_parent &= my_assert(a1_children.size() == 1);
	changing_parent &= my_assert(a1_children[0] == b2);
	changing_parent &= my_assert(b2->get_parent() == a1.get());
	changing_parent &= my_assert(b2->get_children().size() == 0);

	// a2 <-> (b1, b3)
	auto a2_children = a2->get_children();
	changing_parent &= my_assert(a2_children.size() == 2);
	changing_parent &= my_assert(a2_children[0] == b3);
	changing_parent &= my_assert(a2_children[1] == b1);
	changing_parent &= my_assert(b1->get_parent() == a2.get());
	changing_parent &= my_assert(b3->get_parent() == a2.get());

	// (b1, b3) <-> c
	auto b1_children = b1->get_children();
	changing_parent &= my_assert(b1_children.size() == 1);
	changing_parent &= my_assert(b1_children[0] == c);
	changing_parent &= my_assert(c->get_parent() == b1.get());
	changing_parent &= my_assert(b3->get_children().size() == 0);

	// c <-> d
	changing_parent &= my_assert(d->get_parent() == c.get());
	auto c_children = c->get_children();
	changing_parent &= my_assert(c_children.size() == 1);
	changing_parent &= my_assert(c_children[0] == d);
	}


	PVLOG_INFO("Changing parent passed: %d\n", changing_parent);

	//////////////////////////////////////////
	//  Test5 - Changing child
	//////////////////////////////////////////
	b2->add_child(c);

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;


	bool changing_child = true;
	{
	// a1 <-> b2
	auto a1_children = a1->get_children();
	changing_child &= my_assert(a1_children.size() == 1 && a1_children[0] == b2);
	changing_child &= my_assert(b2->get_parent() == a1.get());

	// b2 <-> c
	auto b2_children = b2->get_children();
	changing_child &= my_assert(b2_children.size() == 1 && b2_children[0] == c);
	changing_child &= my_assert(c->get_parent() == b2.get());

	// c <-> d
	auto c_children = c->get_children();
	changing_child &= my_assert(c_children.size() == 1 && c_children[0] == d);
	changing_child &= my_assert(d->get_parent() == c.get());

	// a2 <-> (b1, b3)
	auto a2_children = a2->get_children();
	changing_child &= my_assert(a2_children.size() == 2 && a2_children[0] == b3 && a2_children[1] == b1);
	changing_child &= my_assert(b1->get_parent() == a2.get());
	changing_child &= my_assert(b3->get_parent() == a2.get());
	changing_child &= my_assert(b1->get_children().size() == 0);
	changing_child &= my_assert(b3->get_children().size() == 0);
	}

	PVLOG_INFO("Changing child passed: %d\n", changing_child);

	//////////////////////////////////////////
	//  Test6 - Create with parent and set same parent
	//////////////////////////////////////////
	PVCore::PVDataTreeAutoShared<C> c2(b2);

	bool create_with_parent_and_set_same_parent = true;
	{
		// a1 <-> b2
		auto a1_children = a1->get_children();
		create_with_parent_and_set_same_parent &= my_assert(a1_children.size() == 1 && a1_children[0] == b2);

		// b2 <-> (c, c2)
		create_with_parent_and_set_same_parent &= my_assert(b2->get_parent() == a1.get());
		auto b2_children = b2->get_children();
		create_with_parent_and_set_same_parent &= my_assert(b2_children.size() == 2 && b2_children[0] == c && b2_children[1] == c2);
		create_with_parent_and_set_same_parent &= my_assert(c->get_parent() == b2.get());
		create_with_parent_and_set_same_parent &= my_assert(c2->get_parent() == b2.get());
		create_with_parent_and_set_same_parent &= my_assert(c2->get_children().size() == 0);

		// c <-> d
		create_with_parent_and_set_same_parent &= my_assert(d->get_parent() == c.get());
		auto c_children = c->get_children();
		create_with_parent_and_set_same_parent &= my_assert(c_children.size() == 1 && c_children[0] == d);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	PVLOG_INFO("Create with parent and set same parent: %d\n", create_with_parent_and_set_same_parent);

	//////////////////////////////////////////
	//  Test7 - Remove child
	//////////////////////////////////////////

	a1->remove_child(b2);

	bool removing_child = true;
	{
		removing_child &= my_assert(a1->get_children().size() == 0);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	PVLOG_INFO("Removing child passed: %d\n", removing_child);

	bool removing_parent = true;
#if 0
	//////////////////////////////////////////
	//  Test8 - Remove parent
	//////////////////////////////////////////
	b1->set_parent(nullptr);

	bool removing_parent = true;
	{
		removing_parent &= my_assert(a2->get_parent() == nullptr);
		auto a2_children = a2->get_children();
		removing_parent &= my_assert(a2_children.size() == 1 && a2_children[0] == b3);
		removing_parent &= my_assert(b3->get_parent() == a2.get());
		removing_parent &= my_assert(b3->get_children().size() == 0);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	PVLOG_INFO("Removing parent passed: %d\n", removing_parent);
#endif

	//////////////////////////////////////////
	//  Test9 - Create an object without parent and then set a parent
	//////////////////////////////////////////
#if 0
	boost::shared_ptr<C> c3(new C());
	c3->set_parent(b3);

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	bool create_without_parent_and_set_parent = true;
	{
		create_without_parent_and_set_parent &= my_assert(c3->get_parent() == b3.get());
		auto b3_children = b3->get_children();
		create_without_parent_and_set_parent &= my_assert(b3_children.size() == 1 && b3_children[0] == c3);
	}

	PVLOG_INFO("Create without parent and set parent passed: %d\n", create_without_parent_and_set_parent);
#endif
	//////////////////////////////////////////
	//  Test10 - Get null ancestor if parent is null
	//////////////////////////////////////////
	D_p d3;

	bool null_parent_null_ancestor = true;
	null_parent_null_ancestor &= my_assert(d3->get_parent() == nullptr && d3->get_parent<A>() == nullptr);

	PVLOG_INFO("Get null ancestor if parent is null: %d\n", null_parent_null_ancestor);

	// Delete the remaining hierarchy
	std::cout << std::endl << "=DELETING REMAINING TREES=" << std::endl;
	a1.reset();
	a2.reset();

	return (parent_access && children_access && same_parent && changing_parent && changing_child && create_with_parent_and_set_same_parent && removing_child && removing_parent && null_parent_null_ancestor);
}

bool serialize_use_case()
{
	std::cout << std::endl << std::endl;

	// Serialize datatree
	{
	A_p a1(5);
	B_p b1(a1, 4);
	B_p b2(a1, 3);
	C_p c(b1, 2);
	c->set_j(c->get_j() * 10);
	D_p d(c, 1);
	d->set_j(d->get_j() * 10);
	std::cout << "b1 = " << b1.get() << std::endl;
	std::cout << "b2 = " << b2.get() << std::endl;

	a1->dump();

	a1->save_to_file("datatree_serialized");
	}

	std::cout << std::endl << std::endl;
	// Deserialize datatree
	A_p a1;
	a1->load_from_file("datatree_serialized");

	a1->dump();
	std::cout << std::endl;

	bool hierarchical_serialization = true;
	bool class_content = false;
	bool content_from_parent = false;
	auto a1_children = a1->get_children();
	hierarchical_serialization &= a1_children.size() == 2;
	if (hierarchical_serialization) {
		auto b1 = a1_children[0];
		auto b2 = a1_children[1];
		auto b1_children = b1->get_children();
		auto b2_children = b2->get_children();
		hierarchical_serialization &= b1_children.size() == 1;
		if (hierarchical_serialization) {
			auto c = b1_children[0];
			auto c_children = c->get_children();
			hierarchical_serialization &= c_children.size() == 1;
			if (hierarchical_serialization) {
				auto d = c_children[0];

				// a1 <-> (b1, b2)
				hierarchical_serialization &= my_assert(a1_children.size() == 2 && a1_children[0] == b1 && a1_children[1] == b2);
				hierarchical_serialization &= my_assert(b1->get_parent() == a1.get() && b2->get_parent() == a1.get());
				hierarchical_serialization &= my_assert(b2_children.size() == 0);

				// b1 <-> c
				hierarchical_serialization &= my_assert(b1->get_parent() == a1.get());
				hierarchical_serialization &= my_assert(b1_children.size() == 1 && b1_children[0] == c);
				hierarchical_serialization &= my_assert(c->get_parent() == b1.get());

				// c <-> d
				hierarchical_serialization &= my_assert(d->get_parent() == c.get());
				hierarchical_serialization &= my_assert(c_children.size() == 1 && c_children[0] == d);

				if (hierarchical_serialization) {

					class_content = true;
					class_content &= my_assert(a1->get_i() == 5);
					class_content &= my_assert(b1->get_i() == 4 && b2->get_i() == 3);
					class_content &= my_assert(c->get_i() == 2);
					class_content &= my_assert(d->get_i() == 1);

					if (class_content) {

						content_from_parent = true;
						content_from_parent &= my_assert(b1->get_j() == b1->get_parent()->get_i()*2);
						content_from_parent &= my_assert(b2->get_j() == b2->get_parent()->get_i()*2);
						content_from_parent &= my_assert(c->get_j() == c->get_parent()->get_i()*2*10);
						content_from_parent &= my_assert(d->get_j() == d->get_parent()->get_i()*2*10);
					}
				}
			}
		}
	}

	PVLOG_INFO("Hierarchical serialization/deserialization passed: %d\n", hierarchical_serialization);
	PVLOG_INFO("Class content integrity passed: %d\n", class_content);
	PVLOG_INFO("Class content from parent integrity passed: %d\n", content_from_parent);

	return hierarchical_serialization && class_content && content_from_parent;
}

/******************************************************************************
 *
 * Test case
 *
 *****************************************************************************/

int main()
{
	return !(standard_use_case() && serialize_use_case());
}
