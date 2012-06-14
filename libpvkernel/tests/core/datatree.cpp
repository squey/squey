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
	A(PVCore::PVDataTreeNoParent<A>* parent = NULL) : data_tree_a_t(parent) {};
};

typedef typename PVCore::PVDataTreeObject<A, C> data_tree_b_t;
class B : public data_tree_b_t
{
public:
	B(A* parent = NULL) : data_tree_b_t(parent) {};
};

typedef typename PVCore::PVDataTreeObject<B, D> data_tree_c_t;
class C : public data_tree_c_t
{
public:
	C(B* parent = NULL) : data_tree_c_t(parent) {};
};

typedef typename PVCore::PVDataTreeObject<C, PVCore::PVDataTreeNoChildren<D>> data_tree_d_t;
class D : public data_tree_d_t
{
public:
	D(C* parent = NULL) : data_tree_d_t(parent) {};
};

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
	A* a1 = new A();
	A* a2 = new A();
	B* b1 = new B(a1);
	B* b2 = new B(a1);
	B* b3 = new B(a2);
	C* c = new C(b1);
	D* d = new D(c);

	std::cout << "a1=" << a1 << std::endl;
	std::cout << "a2=" << a2 << std::endl;
	std::cout << "b1=" << b1 << std::endl;
	std::cout << "b2=" << b2 << std::endl;
	std::cout << "b3=" << b3 << std::endl;
	std::cout << "c=" << c << std::endl;
	std::cout << "d=" << d << std::endl;

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump();
	std::cout << std::endl;

	bool parent_access = true;
	parent_access &= (a1->get_parent() == nullptr);
	parent_access &= (a2->get_parent() == nullptr);
	parent_access &= (b1->get_parent() == a1);
	parent_access &= (b2->get_parent() == a1);
	parent_access &= (b3->get_parent() == a2);
	parent_access &= (c->get_parent() == b1);
	parent_access &= (c->get_parent<A>() == a1);
	parent_access &= (d->get_parent() == c);
	parent_access &= (d->get_parent<B>() == b1);
	parent_access &= (d->get_parent<A>() == a1);
	parent_access &= (d->get_parent<C>() == c);
	PVLOG_INFO("Parent access passed: %d\n", parent_access);


	//////////////////////////////////////////
	//  Test2 - Children access
	//////////////////////////////////////////
	bool children_access = true;
	{
	auto a1_children = a1->get_children();
	children_access &= (a1_children.size() == 2 && a1_children[0].get() == b1 && a1_children[1].get() == b2);
	auto a2_children = a2->get_children();
	children_access &= (a2_children.size() == 1 && a2_children[0].get() == b3);
	auto b1_children = b1->get_children();
	children_access &= (b1_children.size() == 1 && b1_children[0].get() == c);
	children_access &= (b2->get_children().size() == 0);
	children_access &= (b3->get_children().size() == 0);
	auto c_children = c->get_children();
	children_access &= (c_children.size() == 1 && c_children[0].get() == d);
	children_access &= (d->get_children().size() == 0);
	}
	PVLOG_INFO("Children access passed: %d\n", children_access);


	//////////////////////////////////////////
	//  Test3 - Same parent
	//////////////////////////////////////////
	bool same_parent = true;
	a1->set_parent(nullptr);
	a2->set_parent(nullptr);
	b1->set_parent(a1);
	b1->set_parent(a1);
	b1->set_parent(a1);
	b3->set_parent(a2);
	c->set_parent(b1);
	c->set_parent(b1);
	d->set_parent(c);

	{
	same_parent &= (a1->get_parent() == nullptr);
	same_parent &= (a2->get_parent() == nullptr);
	same_parent &= (b1->get_parent() == a1);
	same_parent &= (b2->get_parent() == a1);
	same_parent &= (b3->get_parent() == a2);
	same_parent &= (c->get_parent() == b1);
	same_parent &= (c->get_parent<A>() == a1);
	same_parent &= (d->get_parent() == c);
	same_parent &= (d->get_parent<B>() == b1);
	same_parent &= (d->get_parent<A>() == a1);
	same_parent &= (d->get_parent<C>() == c);
	auto a1_children = a1->get_children();
	same_parent &= (a1_children.size() == 2 && a1_children[0].get() == b1 && a1_children[1].get() == b2);
	auto a2_children = a2->get_children();
	same_parent &= (a2_children.size() == 1 && a2_children[0].get() == b3);
	auto b1_children = b1->get_children();
	same_parent &= (b1_children.size() == 1 && b1_children[0].get() == c);
	same_parent &= (b2->get_children().size() == 0);
	same_parent &= (b3->get_children().size() == 0);
	auto c_children = c->get_children();
	same_parent &= (c_children.size() == 1 && c_children[0].get() == d);
	same_parent &= (d->get_children().size() == 0);
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
	changing_parent &= (a1->get_parent() == nullptr);
	auto a1_children = a1->get_children();
	changing_parent &= (a1_children.size() == 1);
	changing_parent &= (a1_children[0].get() == b2);
	changing_parent &= (b2->get_parent() == a1);
	changing_parent &= (b2->get_children().size() == 0);

	// a2 <-> (b1, b3)
	changing_parent &= (a2->get_parent() == nullptr);
	auto a2_children = a2->get_children();
	changing_parent &= (a2_children.size() == 2);
	changing_parent &= (a2_children[0].get() == b3);
	changing_parent &= (a2_children[1].get() == b1);
	changing_parent &= (b1->get_parent() == a2);
	changing_parent &= (b3->get_parent() == a2);

	// (b1, b3) <-> c
	auto b1_children = b1->get_children();
	changing_parent &= (b1_children.size() == 1);
	changing_parent &= (b1_children[0].get() == c);
	changing_parent &= (c->get_parent() == b1);
	changing_parent &= (b3->get_children().size() == 0);

	// c <-> d
	changing_parent &= (d->get_parent() == c);
	auto c_children = c->get_children();
	changing_parent &= (c_children.size() == 1);
	changing_parent &= (c_children[0].get() == d);
	changing_parent &= (d->get_children().size() == 0);
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
	changing_child &= (a1->get_parent() == nullptr);
	auto a1_children = a1->get_children();
	changing_child &= (a1_children.size() == 1 && a1_children[0].get() == b2);
	changing_child &= (b2->get_parent() == a1);

	// b2 <-> c
	auto b2_children = b2->get_children();
	changing_child &= (b2_children.size() == 1 && b2_children[0].get() == c);
	changing_child &= (c->get_parent() == b2);

	// c <-> d
	auto c_children = c->get_children();
	changing_child &= (c_children.size() == 1 && c_children[0].get() == d);
	changing_child &= (d->get_parent() == c);

	// a2 <-> (b1, b3)
	auto a2_children = a2->get_children();
	changing_child &= (a2_children.size() == 2 && a2_children[0].get() == b3 && a2_children[1].get() == b1);
	changing_child &= (b1->get_parent() == a2);
	changing_child &= (b3->get_parent() == a2);
	changing_child &= (b1->get_children().size() == 0);
	changing_child &= (b3->get_children().size() == 0);
	}

	PVLOG_INFO("Changing child passed: %d\n", changing_child);

	//////////////////////////////////////////
	//  Test6 - Create with parent and set same parent
	//////////////////////////////////////////
	C* c2 = new C(b2);
	c2->set_parent(b2);

	bool create_with_parent_and_set_same_parent = true;
	{
		// a1 <-> b2
		create_with_parent_and_set_same_parent &= (a1->get_parent() == nullptr);
		auto a1_children = a1->get_children();
		create_with_parent_and_set_same_parent &= (a1_children.size() == 1 && a1_children[0].get() == b2);

		// b2 <-> (c, c2)
		create_with_parent_and_set_same_parent &= (b2->get_parent() == a1);
		auto b2_children = b2->get_children();
		create_with_parent_and_set_same_parent &= (b2_children.size() == 2 && b2_children[0].get() == c && b2_children[1].get() == c2);
		create_with_parent_and_set_same_parent &= (c->get_parent() == b2);
		create_with_parent_and_set_same_parent &= (c2->get_parent() == b2);
		create_with_parent_and_set_same_parent &= (c2->get_children().size() == 0);

		// c <-> d
		create_with_parent_and_set_same_parent &= (d->get_parent() == c);
		auto c_children = c->get_children();
		create_with_parent_and_set_same_parent &= (c_children.size() == 1 && c_children[0].get() == d);
		create_with_parent_and_set_same_parent &= (d->get_children().size() == 0);
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
		removing_child &= (a1->get_parent() == nullptr);
		removing_child &= (a1->get_children().size() == 0);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	PVLOG_INFO("Removing child passed: %d\n", removing_child);

	//////////////////////////////////////////
	//  Test8 - Remove parent
	//////////////////////////////////////////
	b1->set_parent(nullptr);

	bool removing_parent = true;
	{
		removing_parent &= (a2->get_parent() == nullptr);
		auto a2_children = a2->get_children();
		removing_parent &= (a2_children.size() == 1 && a2_children[0].get() == b3);
		removing_parent &= (b3->get_parent() == a2);
		removing_parent &= (b3->get_children().size() == 0);
	}

	std::cout << "a1:" << std::endl;
	a1->dump();
	std::cout << "a2:" << std::endl;
	a2->dump(); std::cout << std::endl;

	//////////////////////////////////////////
	//  Test8 - Get null ancestor if parent is null
	//////////////////////////////////////////
	D d2;

	bool null_parent_null_ancestor = true;
	null_parent_null_ancestor &= (d2.get_parent() == nullptr && d2.get_parent<A>() == nullptr);

	PVLOG_INFO("Get null ancestor if parent is null: %d\n", null_parent_null_ancestor);

	// Delete the remaining hierarchy
	std::cout << std::endl << "=DELETING REMAINING TREES=" << std::endl;
	delete a1;
	delete a2;

	return !(parent_access && children_access && same_parent && changing_parent && changing_child && removing_child && removing_parent && null_parent_null_ancestor);
}
