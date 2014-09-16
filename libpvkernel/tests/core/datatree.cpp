/**
 * \file datatree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>

#include <pvkernel/core/picviz_assert.h>

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
		so.attribute("_i", _i);
		so.attribute("_j", _j);
	}

	virtual void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
	{
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

void delete_use_case()
{
	std::cout << "Delete use case" << std::endl;
	A_p a1(4);
	{
		B_p b1(a1, 5);
		B_p b2(a1);
		C_p c(b1);
		D_p d(c);
	}

	a1.reset();
}

void standard_use_case()
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
	PV_ASSERT_VALID(b1->a_was_here());
	PV_ASSERT_VALID(b2->a_was_here());
	PV_ASSERT_VALID(b3->a_was_here());

	// Auto shared check
	B_p b1_other = b1;
	std::cout << "B1 other: " << b1_other.get() << " , b1: " << b1.get() << std::endl;
	PV_ASSERT_VALID(b1_other.get() == b1.get());
	PV_ASSERT_VALID(b1_other->i() == 5);
	b1_other = b2;
	std::cout << "B2 other: " << b1_other.get() << " , b2: " << b2.get() << std::endl;
	PV_ASSERT_VALID(b1_other.get() == b2.get());
	PV_ASSERT_VALID(b1_other->i() == 0);


	// Check invalid pointers
	C_p c_inv = C_p::invalid();
	PV_ASSERT_VALID(c_inv.get() == nullptr);

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

	PV_ASSERT_VALID(b1->get_parent() == a1.get());
	PV_ASSERT_VALID(b2->get_parent() == a1.get());
	PV_ASSERT_VALID(b3->get_parent() == a2.get());
	PV_ASSERT_VALID(c->get_parent() == b1.get());
	PV_ASSERT_VALID(c->get_parent<A>() == a1.get());
	PV_ASSERT_VALID(d->get_parent() == c.get());
	PV_ASSERT_VALID(d->get_parent<B>() == b1.get());
	PV_ASSERT_VALID(d->get_parent<A>() == a1.get());
	PV_ASSERT_VALID(d->get_parent<C>() == c.get());
	PVLOG_INFO("Parent access passed\n");


	//////////////////////////////////////////
	//  Test2 - Children access
	//////////////////////////////////////////
	{
		auto a1_children = a1->get_children();
		PV_ASSERT_VALID(a1_children.size() == 2 && a1_children[0] == b1 && a1_children[1] == b2);
		auto a2_children = a2->get_children();
		PV_ASSERT_VALID(a2_children.size() == 1 && a2_children[0] == b3);
		auto b1_children = b1->get_children();
		PV_ASSERT_VALID(b1_children.size() == 1 && b1_children[0] == c);
		PV_ASSERT_VALID(b2->get_children().size() == 0);
		PV_ASSERT_VALID(b3->get_children().size() == 0);
		auto c_children = c->get_children();
		PV_ASSERT_VALID(c_children.size() == 1 && c_children[0] == d);
	}
	PVLOG_INFO("Children access passed\n");


	//////////////////////////////////////////
	//  Test3 - Same parent
	//////////////////////////////////////////
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
		//PV_ASSERT_VALID(a1->get_parent() == nullptr);
		//PV_ASSERT_VALID(a2->get_parent() == nullptr);
		PV_ASSERT_VALID(b1->get_parent() == a1.get());
		PV_ASSERT_VALID(b2->get_parent() == a1.get());
		PV_ASSERT_VALID(b3->get_parent() == a2.get());
		PV_ASSERT_VALID(c->get_parent() == b1.get());
		PV_ASSERT_VALID(c->get_parent<A>() == a1.get());
		PV_ASSERT_VALID(d->get_parent() == c.get());
		PV_ASSERT_VALID(d->get_parent<B>() == b1.get());
		PV_ASSERT_VALID(d->get_parent<A>() == a1.get());
		PV_ASSERT_VALID(d->get_parent<C>() == c.get());
		auto a1_children = a1->get_children();
		PV_ASSERT_VALID(a1_children.size() == 2 && a1_children[0] == b1 && a1_children[1] == b2);
		auto a2_children = a2->get_children();
		PV_ASSERT_VALID(a2_children.size() == 1 && a2_children[0] == b3);
		auto b1_children = b1->get_children();
		PV_ASSERT_VALID(b1_children.size() == 1 && b1_children[0] == c);
		PV_ASSERT_VALID(b2->get_children().size() == 0);
		PV_ASSERT_VALID(b3->get_children().size() == 0);
		auto c_children = c->get_children();
		PV_ASSERT_VALID(c_children.size() == 1 && c_children[0] == d);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;


	PVLOG_INFO("Reaffecting same parent passed\n");

	//////////////////////////////////////////
	//  Test4 - Changing parent
	//////////////////////////////////////////
	b1->set_parent(a2);

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;


	{
		// a1 <-> b2
		auto a1_children = a1->get_children();
		PV_ASSERT_VALID(a1_children.size() == 1);
		PV_ASSERT_VALID(a1_children[0] == b2);
		PV_ASSERT_VALID(b2->get_parent() == a1.get());
		PV_ASSERT_VALID(b2->get_children().size() == 0);

		// a2 <-> (b1, b3)
		auto a2_children = a2->get_children();
		PV_ASSERT_VALID(a2_children.size() == 2);
		PV_ASSERT_VALID(a2_children[0] == b3);
		PV_ASSERT_VALID(a2_children[1] == b1);
		PV_ASSERT_VALID(b1->get_parent() == a2.get());
		PV_ASSERT_VALID(b3->get_parent() == a2.get());

		// (b1, b3) <-> c
		auto b1_children = b1->get_children();
		PV_ASSERT_VALID(b1_children.size() == 1);
		PV_ASSERT_VALID(b1_children[0] == c);
		PV_ASSERT_VALID(c->get_parent() == b1.get());
		PV_ASSERT_VALID(b3->get_children().size() == 0);

		// c <-> d
		PV_ASSERT_VALID(d->get_parent() == c.get());
		auto c_children = c->get_children();
		PV_ASSERT_VALID(c_children.size() == 1);
		PV_ASSERT_VALID(c_children[0] == d);
	}
	PVLOG_INFO("Changing parent passed\n");

	//////////////////////////////////////////
	//  Test5 - Changing child
	//////////////////////////////////////////
	b2->add_child(c);

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	{
		// a1 <-> b2
		auto a1_children = a1->get_children();
		PV_ASSERT_VALID(a1_children.size() == 1 && a1_children[0] == b2);
		PV_ASSERT_VALID(b2->get_parent() == a1.get());

		// b2 <-> c
		auto b2_children = b2->get_children();
		PV_ASSERT_VALID(b2_children.size() == 1 && b2_children[0] == c);
		PV_ASSERT_VALID(c->get_parent() == b2.get());

		// c <-> d
		auto c_children = c->get_children();
		PV_ASSERT_VALID(c_children.size() == 1 && c_children[0] == d);
		PV_ASSERT_VALID(d->get_parent() == c.get());

		// a2 <-> (b1, b3)
		auto a2_children = a2->get_children();
		PV_ASSERT_VALID(a2_children.size() == 2 && a2_children[0] == b3 && a2_children[1] == b1);
		PV_ASSERT_VALID(b1->get_parent() == a2.get());
		PV_ASSERT_VALID(b3->get_parent() == a2.get());
		PV_ASSERT_VALID(b1->get_children().size() == 0);
		PV_ASSERT_VALID(b3->get_children().size() == 0);
	}

	PVLOG_INFO("Changing child passed\n");

	//////////////////////////////////////////
	//  Test6 - Create with parent and set same parent
	//////////////////////////////////////////
	PVCore::PVDataTreeAutoShared<C> c2(b2);

	{
		// a1 <-> b2
		auto a1_children = a1->get_children();
		PV_ASSERT_VALID(a1_children.size() == 1 && a1_children[0] == b2);

		// b2 <-> (c, c2)
		PV_ASSERT_VALID(b2->get_parent() == a1.get());
		auto b2_children = b2->get_children();
		PV_ASSERT_VALID(b2_children.size() == 2 && b2_children[0] == c && b2_children[1] == c2);
		PV_ASSERT_VALID(c->get_parent() == b2.get());
		PV_ASSERT_VALID(c2->get_parent() == b2.get());
		PV_ASSERT_VALID(c2->get_children().size() == 0);

		// c <-> d
		PV_ASSERT_VALID(d->get_parent() == c.get());
		auto c_children = c->get_children();
		PV_ASSERT_VALID(c_children.size() == 1 && c_children[0] == d);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	PVLOG_INFO("Create with parent and set same parent passed\n");

	//////////////////////////////////////////
	//  Test7 - Remove child
	//////////////////////////////////////////

	a1->remove_child(b2);

	{
		PV_ASSERT_VALID(a1->get_children().size() == 0);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	PVLOG_INFO("Removing child passed\n");

#if 0
	//////////////////////////////////////////
	//  Test8 - Remove parent
	//////////////////////////////////////////
	b1->set_parent(nullptr);

	{
		PV_ASSERT_VALID(a2->get_parent() == nullptr);
		auto a2_children = a2->get_children();
		PV_ASSERT_VALID(a2_children.size() == 1 && a2_children[0] == b3);
		PV_ASSERT_VALID(b3->get_parent() == a2.get());
		PV_ASSERT_VALID(b3->get_children().size() == 0);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	PVLOG_INFO("Removing parent passed\n");
#endif

	//////////////////////////////////////////
	//  Test9 - Create an object without parent and then set a parent
	//////////////////////////////////////////
#if 0
	std::shared_ptr<C> c3(new C());
	c3->set_parent(b3);

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	{
		PV_ASSERT_VALID(c3->get_parent() == b3.get());
		auto b3_children = b3->get_children();
		PV_ASSERT_VALID(b3_children.size() == 1 && b3_children[0] == c3);
	}

	PVLOG_INFO("Create without parent and set parent passed\n");
#endif
	//////////////////////////////////////////
	//  Test10 - Get null ancestor if parent is null
	//////////////////////////////////////////
	D_p d3;

	PV_ASSERT_VALID(d3->get_parent() == nullptr && d3->get_parent<A>() == nullptr);

	PVLOG_INFO("Get null ancestor if parent is null passed\n");

	//////////////////////////////////////////
	//  Test11 - Start from a PVDataTreeObjectBase and check its property
	//////////////////////////////////////////

	PVCore::PVDataTreeObjectBase* obase = static_cast<PVCore::PVDataTreeObjectBase*>(a1.get());
	PVCore::PVDataTreeObjectBase::base_p_type obase_p = obase->base_shared_from_this();
	PVLOG_INFO("a1=%p obase=%p obase_get=%p\n", a1.get(), obase, obase_p.get());

	PV_ASSERT_VALID(obase_p.get() == obase);
	PV_ASSERT_VALID(dynamic_cast<PVCore::PVDataTreeObjectWithParentBase*>(obase) == nullptr);
	PV_ASSERT_VALID(dynamic_cast<PVCore::PVDataTreeObjectWithChildrenBase*>(obase) != nullptr);

	obase = static_cast<PVCore::PVDataTreeObjectBase*>(b1.get());
	PV_ASSERT_VALID(dynamic_cast<PVCore::PVDataTreeObjectWithParentBase*>(obase) != nullptr);
	PV_ASSERT_VALID(dynamic_cast<PVCore::PVDataTreeObjectWithChildrenBase*>(obase) != nullptr);

	obase = static_cast<PVCore::PVDataTreeObjectBase*>(d.get());
	PV_ASSERT_VALID(dynamic_cast<PVCore::PVDataTreeObjectWithParentBase*>(obase) != nullptr);
	PV_ASSERT_VALID(dynamic_cast<PVCore::PVDataTreeObjectWithChildrenBase*>(obase) == nullptr);

	//////////////////////////////////////////
	//  Test11 - Start from a PVDataTreeObjectBase and check that casting works
	//////////////////////////////////////////
	// obase is 'd' here
	std::cout << "Base object pointer = " << obase << std::endl;
	std::cout << "Starting address of final object = " << PVCore::PVTypeTraits::get_starting_address(obase) << std::endl;
	std::cout << "Final object (D) pointer is = " << d.get() << std::endl;
	PV_ASSERT_VALID(PVCore::PVTypeTraits::get_starting_address(obase) == d.get());

	// Delete the remaining hierarchy
	std::cout << std::endl << "=DELETING REMAINING TREES=" << std::endl;
	a1.reset();
	a2.reset();
}

void serialize_use_case()
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

	auto a1_children = a1->get_children();

	PV_ASSERT_VALID(a1_children.size() == 2);

	auto b1 = a1_children[0];
	auto b2 = a1_children[1];
	auto b1_children = b1->get_children();
	auto b2_children = b2->get_children();

	PV_ASSERT_VALID(b1_children.size() == 1);

	auto c = b1_children[0];
	auto c_children = c->get_children();

	PV_ASSERT_VALID(c_children.size() == 1);

	auto d = c_children[0];

	// a1 <-> (b1, b2)
	PV_ASSERT_VALID(a1_children.size() == 2 && a1_children[0] == b1 && a1_children[1] == b2);
	PV_ASSERT_VALID(b1->get_parent() == a1.get() && b2->get_parent() == a1.get());
	PV_ASSERT_VALID(b2_children.size() == 0);

	// b1 <-> c
	PV_ASSERT_VALID(b1->get_parent() == a1.get());
	PV_ASSERT_VALID(b1_children.size() == 1 && b1_children[0] == c);
	PV_ASSERT_VALID(c->get_parent() == b1.get());

	// c <-> d
	PV_ASSERT_VALID(d->get_parent() == c.get());
	PV_ASSERT_VALID(c_children.size() == 1 && c_children[0] == d);

	PV_ASSERT_VALID(a1->get_i() == 5);
	PV_ASSERT_VALID(b1->get_i() == 4 && b2->get_i() == 3);
	PV_ASSERT_VALID(c->get_i() == 2);
	PV_ASSERT_VALID(d->get_i() == 1);

	PV_ASSERT_VALID(b1->get_j() == b1->get_parent()->get_i()*2);
	PV_ASSERT_VALID(b2->get_j() == b2->get_parent()->get_i()*2);
	PV_ASSERT_VALID(c->get_j() == c->get_parent()->get_i()*2*10);
	PV_ASSERT_VALID(d->get_j() == d->get_parent()->get_i()*2*10);

	PVLOG_INFO("Class content integrity passed\n");
	PVLOG_INFO("Class content from parent integrity passed\n");

	PVLOG_INFO("Hierarchical serialization/deserialization passed\n");
}

/******************************************************************************
 *
 * Test case
 *
 *****************************************************************************/

int main()
{

	standard_use_case();
	serialize_use_case();
	delete_use_case();

	return 0;
}
