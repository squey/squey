/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>
//#include <pvguiqt/PVHiveDataTreeModel.h>

#include <QApplication>
#include <QMainWindow>
#include <QDialog>
#include <QVBoxLayout>
#include <QTreeView>

#include <boost/thread.hpp>

#include <unistd.h> // for usleep

// Data-tree structure
//

// forward declarations
class A;
class B;
class C;
class D;

class D : public PVCore::PVDataTreeChild<C, D>
{

  public:
	D(C* c, int i = 0) : PVCore::PVDataTreeChild<C, D>(c), _i(i) {}

  public:
	virtual ~D() { std::cout << "~D(" << this << ")" << std::endl; }

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

  private:
	int _i;
};

class C : public PVCore::PVDataTreeChild<B, C>, public PVCore::PVDataTreeParent<D, C>
{

  public:
	C(B* b, int i = 0) : PVCore::PVDataTreeChild<B, C>(b), _i(i) {}

  public:
	virtual ~C() { std::cout << "~C(" << this << ")" << std::endl; }

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

  private:
	int _i;
};

class B : public PVCore::PVDataTreeChild<A, B>, public PVCore::PVDataTreeParent<C, B>
{
	friend class A;

  public:
	B(A* a, int i = 0) : PVCore::PVDataTreeChild<A, B>(a), _i(i) {}

  public:
	virtual ~B() { std::cout << "~B(" << this << ")" << std::endl; }

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

  private:
	int _i;
};

class A : public PVCore::PVDataTreeParent<B, A>
{

  public:
	A(int i = 0) : PVCore::PVDataTreeParent<B, A>(), _i(i) {}

  public:
	virtual ~A() { std::cout << "~A(" << this << ")" << std::endl; }

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

  private:
	int _i;
};

using A_p = PVCore::PVSharedPtr<A>;
using B_p = PVCore::PVSharedPtr<B>;
using C_p = PVCore::PVSharedPtr<C>;
using D_p = PVCore::PVSharedPtr<D>;

int main(/*int argc, char** argv*/)
{
	// Objects, let's create our tree !
	A_p a(new A());
	B_p b1 = a->emplace_add_child(0);
	B_p b2 = a->emplace_add_child(1);
	C_p c1 = b1->emplace_add_child(0);
	C_p c2 = b1->emplace_add_child(1);
	C_p c4 = b2->emplace_add_child(2);
	C_p c5 = b2->emplace_add_child(3);
	D_p d1 = c1->emplace_add_child(0);
	D_p d2 = c1->emplace_add_child(1);
	D_p d4 = c2->emplace_add_child(2);
	D_p d5 = c2->emplace_add_child(3);
	D_p d6 = c4->emplace_add_child(4);
	D_p d7 = c5->emplace_add_child(5);

	//// Qt app
	// QApplication app(argc, argv);

	//// Create our model and view
	// PVGuiQt::PVHiveDataTreeModel* model = new PVGuiQt::PVHiveDataTreeModel(*a);
	// QTreeView* view = new QTreeView();
	// view->setModel(model);

	// QMainWindow* mw = new QMainWindow();
	// mw->setCentralWidget(view);

	// mw->show();

	// Boost thread that changes values
	boost::thread th([&] {
		PVHive::PVActor<B> actor_b1;
		PVHive::PVActor<B> actor_b2;
		PVHive::PVActor<C> actor_c1;
		PVHive::PVActor<D> actor_d2;
		PVHive::PVActor<D> actor_d7;
		PVHive::get().register_actor(b1, actor_b1);
		PVHive::get().register_actor(b2, actor_b2);
		PVHive::get().register_actor(c1, actor_c1);
		PVHive::get().register_actor(d2, actor_d2);
		PVHive::get().register_actor(d7, actor_d7);
		int i_b = 0;
		int i_c = 1;
		int i_d = 2;
		while (true) {
			actor_b1.call<FUNC(B::set_i)>(i_b);
			actor_b2.call<FUNC(B::set_i)>(i_b);
			actor_c1.call<FUNC(C::set_i)>(i_c);
			actor_d2.call<FUNC(D::set_i)>(i_d);
			actor_d7.call<FUNC(D::set_i)>(i_d);
			i_b++;
			i_c++;
			i_d++;
			usleep(100 * 1000);
		}
	});

	return 0;
	// return app.exec();
}
