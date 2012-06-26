#include <pvkernel/core/PVDataTreeObject.h>
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

typedef typename PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<A>, B> data_tree_a_t;
class A : public data_tree_a_t
{
public:
	A(int i = 0):
		data_tree_a_t(),
		_i(i)
	{ }
private:
	int _i;
};

typedef typename PVCore::PVDataTreeObject<A, C> data_tree_b_t;
class B : public data_tree_b_t
{
	friend class PVCore::PVDataTreeAutoShared<B>;
public:
	virtual ~B() { std::cout << "B " << this << " deleted" << std::endl; }
protected:
	B(int i = 0):
		data_tree_b_t(),
		_i(i)
   	{ }
public:
	void f() const { std::cout << "B i = " << _i << std::endl; }
	inline int i() const { return _i; }
private:
	int _i;
};

typedef typename PVCore::PVDataTreeObject<B, D> data_tree_c_t;
class C : public data_tree_c_t
{
public:
	C():
		data_tree_c_t()
	{ }

	virtual void set_parent_from_ptr(B* parent)
	{
		data_tree_c_t::set_parent_from_ptr(parent);
		std::cout << "C::set_parent is overriden to add special functionnalities." << std::endl;
	}
};

typedef typename PVCore::PVDataTreeObject<C, PVCore::PVDataTreeNoChildren<D>> data_tree_d_t;
class D : public data_tree_d_t
{
public:
	D():
		data_tree_d_t()
   	{ }
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

/******************************************************************************
 *
 * Test case
 *
 *****************************************************************************/
int main()
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
	parent_access &= my_assert(a1->get_parent() == nullptr);
	parent_access &= my_assert(a2->get_parent() == nullptr);
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
	children_access &= my_assert(d->get_children().size() == 0);
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
	same_parent &= my_assert(d->get_children().size() == 0);
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
	changing_parent &= my_assert(a1->get_parent() == nullptr);
	auto a1_children = a1->get_children();
	changing_parent &= my_assert(a1_children.size() == 1);
	changing_parent &= my_assert(a1_children[0] == b2);
	changing_parent &= my_assert(b2->get_parent() == a1.get());
	changing_parent &= my_assert(b2->get_children().size() == 0);

	// a2 <-> (b1, b3)
	changing_parent &= my_assert(a2->get_parent() == nullptr);
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
	changing_parent &= my_assert(d->get_children().size() == 0);
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
	changing_child &= my_assert(a1->get_parent() == nullptr);
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
		create_with_parent_and_set_same_parent &= my_assert(a1->get_parent() == nullptr);
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
		create_with_parent_and_set_same_parent &= my_assert(d->get_children().size() == 0);
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
		removing_child &= my_assert(a1->get_parent() == nullptr);
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

	//////////////////////////////////////////
	//  Test10 - Get null ancestor if parent is null
	//////////////////////////////////////////
	D d3;

	bool null_parent_null_ancestor = true;
	null_parent_null_ancestor &= my_assert(d3.get_parent() == nullptr && d3.get_parent<A>() == nullptr);

	PVLOG_INFO("Get null ancestor if parent is null: %d\n", null_parent_null_ancestor);

	// Delete the remaining hierarchy
	std::cout << std::endl << "=DELETING REMAINING TREES=" << std::endl;
	a1.reset();
	a2.reset();

	return !(parent_access && children_access && same_parent && changing_parent && changing_child && create_with_parent_and_set_same_parent && removing_child && removing_parent && create_without_parent_and_set_parent && null_parent_null_ancestor);
}
