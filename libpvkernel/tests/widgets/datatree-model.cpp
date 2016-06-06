/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/widgets/PVDataTreeModel.h>

#include <QApplication>
#include <QMainWindow>
#include <QDialog>
#include <QVBoxLayout>
#include <QTreeView>

#include <libgen.h>

// Data-tree structure
//

// forward declarations
class A;
class B;
class C;
class D;
class E;

class E : public PVCore::PVDataTreeChild<D, E>
{

  public:
	E(D* d, int i = 0) : PVCore::PVDataTreeChild<D, E>(d), _i(i) {}

  public:
	virtual ~E() { std::cout << "~E(" << this << ")" << std::endl; }

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

  private:
	int _i;
};

class D : public PVCore::PVDataTreeParent<E, D>, public PVCore::PVDataTreeChild<C, D>
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

class C : public PVCore::PVDataTreeParent<D, C>, public PVCore::PVDataTreeChild<B, C>
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

class B : public PVCore::PVDataTreeParent<C, B>, public PVCore::PVDataTreeChild<A, B>
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
using E_p = PVCore::PVSharedPtr<E>;

void usage(char* progname)
{
	std::cerr << "usage: " << basename(progname) << " [1|2|3]" << std::endl;
	std::cerr << "\t1: to display only model view" << std::endl;
}

int main(int argc, char** argv)
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
	E_p e1 = d1->emplace_add_child(0);
	E_p e2 = d1->emplace_add_child(1);

	// Qt app
	QApplication app(argc, argv);

	int what = 1;

	if (argc != 1) {
		what = atoi(argv[1]);
	}
	if (what == 0) {
		usage(argv[0]);
		return 1;
	}

	return 0;
	//// Create our model and its view
	// PVWidgets::PVDataTreeModel* model = new PVWidgets::PVDataTreeModel(*a);

	// QTreeView* view = new QTreeView();
	// view->setModel(model);
	// view->expandAll();

	// QMainWindow* mw = new QMainWindow();
	// mw->setCentralWidget(view);
	// mw->setWindowTitle("Data Tree - Model");
	// mw->show();

	// return app.exec();
}
