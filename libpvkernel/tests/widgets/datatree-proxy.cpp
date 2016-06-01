/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/widgets/PVDataTreeModel.h>

#include <pvkernel/widgets/PVDataTreeMaskProxyModel.h>

#include <QApplication>
#include <QMainWindow>
#include <QDialog>
#include <QVBoxLayout>
#include <QTreeView>

// Data-tree structure
//

// forward declarations
class A;
class B;
class C;
class D;

typedef typename PVCore::PVDataTreeObject<C, PVCore::PVDataTreeNoChildren<D>> data_tree_d_t;
class D : public data_tree_d_t
{

  public:
	D(C* c, int i = 0) : data_tree_d_t(c), _i(i) {}

  public:
	virtual ~D() {}

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

	virtual QString get_serialize_description() const override
	{
		return QString("D: ") + QString::number(get_i());
	}

	void serialize(PVCore::PVSerializeObject&, PVCore::PVSerializeArchive::version_t) {}
	void serialize_read(PVCore::PVSerializeObject&) {}
	void serialize_write(PVCore::PVSerializeObject&) {}

  private:
	int _i;
};

typedef typename PVCore::PVDataTreeObject<B, D> data_tree_c_t;
class C : public data_tree_c_t
{

  public:
	C(B* b, int i = 0) : data_tree_c_t(b), _i(i) {}

  public:
	virtual ~C() {}

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

	virtual QString get_serialize_description() const override
	{
		return QString("C: ") + QString::number(get_i());
	}

	void serialize(PVCore::PVSerializeObject&, PVCore::PVSerializeArchive::version_t) {}
	void serialize_read(PVCore::PVSerializeObject&) override {}
	void serialize_write(PVCore::PVSerializeObject&) override {}

  private:
	int _i;
};

typedef typename PVCore::PVDataTreeObject<A, C> data_tree_b_t;
class B : public data_tree_b_t
{
	friend class A;

  public:
	B(A* a, int i = 0) : data_tree_b_t(a), _i(i) {}

  public:
	virtual ~B() {}

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

	virtual QString get_serialize_description() const override
	{
		return QString("B: ") + QString::number(get_i());
	}

	void serialize(PVCore::PVSerializeObject&, PVCore::PVSerializeArchive::version_t) {}
	void serialize_read(PVCore::PVSerializeObject&) override {}
	void serialize_write(PVCore::PVSerializeObject&) override {}

  private:
	int _i;
};

typedef typename PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<A>, B> data_tree_a_t;
class A : public data_tree_a_t
{

  public:
	A(int i = 0) : data_tree_a_t(), _i(i) {}

  public:
	virtual ~A() {}

  public:
	int get_i() const { return _i; }
	void set_i(int i) { _i = i; }

	virtual QString get_serialize_description() const override
	{
		return QString("A: ") + QString::number(get_i());
	}

	void serialize(PVCore::PVSerializeObject&, PVCore::PVSerializeArchive::version_t) {}
	void serialize_read(PVCore::PVSerializeObject&) override {}
	void serialize_write(PVCore::PVSerializeObject&) override {}

  private:
	int _i;
};

typedef typename A::p_type A_p;
typedef typename B::p_type B_p;
typedef typename C::p_type C_p;
typedef typename D::p_type D_p;

typedef PVWidgets::PVDataTreeMaskProxyModel<C> proxy_model_t;

void print_model_index(const QModelIndex& index)
{
	PVCore::PVDataTreeObjectBase* object =
	    static_cast<PVCore::PVDataTreeObjectBase*>(index.internalPointer());

	if (index.isValid() && (object != nullptr)) {
		std::cout << "(" << index.row() << "," << index.column() << ","
		          << qPrintable(object->get_serialize_description()) << ")";
	} else {
		std::cout << "(" << index.row() << "," << index.column() << ",0)";
	}
}

void print_proxy_tree(const QAbstractItemModel& m,
                      const QModelIndex index = QModelIndex(),
                      const int decal = 0)
{
	for (int i = 0; i < decal; ++i) {
		std::cout << "  ";
	}

	print_model_index(index);
	std::cout << std::endl;

	int count = m.rowCount(index);

	for (int i = 0; i < count; ++i) {
		QModelIndex child = m.index(i, 0, index);
		print_proxy_tree(m, child, decal + 1);
	}
}

int main(int argc, char** argv)
{
	// Objects, let's create our tree !

	/* DATATREE      MODEL                  MASK PROXY(C)
	 *
	 * - a           (-1,-1,0)              (-1,-1,0)
	 *   - b1          (0,0,B: 0)             (0,0,B: 0)
	 *     - c1          (0,0,C: 0)             (0,0,D: 0)
	 *       - d1          (0,0,D: 0)           (0,0,D: 1)
	 *       - d2          (0,0,D: 1)           (0,0,D: 2)
	 *     - c2          (1,0,C: 1)             (0,0,D: 3)
	 *       - d4          (0,0,D: 2)         (1,0,B: 1)
	 *       - d5          (0,0,D: 3)           (0,0,D: 4)
	 *   - b2          (1,0,B: 1)               (0,0,D: 5)
	 *     - c4          (0,0,C: 2)
	 *       - d6          (0,0,D: 4)
	 *     - c5          (0,0,C: 3)
	 *       - d7          (0,0,D: 5)
	 */
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

	// Qt app
	QApplication app(argc, argv);

	// Create our model and view
	PVWidgets::PVDataTreeModel* model = new PVWidgets::PVDataTreeModel(*a);

	std::cout << "MODEL" << std::endl;
	print_proxy_tree(*model);

	proxy_model_t* proxy = new proxy_model_t();
	proxy->setSourceModel(model);

	std::cout << "MASK PROXY(C)" << std::endl;
	print_proxy_tree(*proxy);

	QModelIndex proxy_root = QModelIndex();

	if (proxy->rowCount(proxy_root) != 2) {
		std::cerr << "proxy's root has not 2 children" << std::endl;
		return 1;
	}

	QModelIndex proxy_b1 = proxy->index(0, 0, proxy_root);
	QModelIndex parent;

	parent = proxy->parent(proxy_b1);
	if (parent.isValid()) {
		std::cout << "proxy B1's parent is not the root: ";
		print_model_index(parent);
		std::cout << std::endl;
		return 1;
	}

	if (proxy->rowCount(proxy_b1) != 4) {
		std::cerr << "proxy B1 has not 4 children" << std::endl;
		return 1;
	}

	QModelIndex proxy_d1 = proxy->index(0, 0, proxy_b1);

	parent = proxy->parent(proxy_d1);
	if (parent != proxy_b1) {
		std::cout << "proxy D1's parent is not B1";
		print_model_index(parent);
		std::cout << std::endl;
		return 1;
	}

	QModelIndex src_root = QModelIndex();

	QModelIndex src_b1 = model->index(0, 0, src_root);

	QModelIndex tmp = proxy->mapToSource(proxy_b1);
	if (tmp != src_b1) {
		std::cout << "B1: ::mapToSource(";
		print_model_index(proxy_b1);
		std::cout << ") fails: ";
		print_model_index(tmp);
		std::cout << std::endl;
		return 1;
	}

	QModelIndex src_c1 = model->index(0, 0, src_b1);
	QModelIndex src_d1 = model->index(0, 0, src_c1);

	tmp = proxy->mapToSource(proxy_d1);
	if (tmp != src_d1) {
		std::cout << "D1: ::mapToSource(";
		print_model_index(proxy_d1);
		std::cout << ") fails: ";
		print_model_index(tmp);
		std::cout << std::endl;
		return 1;
	}

	tmp = proxy->mapFromSource(src_b1);
	if (tmp != proxy_b1) {
		std::cout << "B1: ::mapFromSource(";
		print_model_index(src_d1);
		std::cout << ") fails: ";
		print_model_index(tmp);
		std::cout << std::endl;
		return 1;
	}

	tmp = proxy->mapFromSource(src_d1);
	if (tmp != proxy_d1) {
		std::cout << "D1: ::mapFromSource(";
		print_model_index(src_d1);
		std::cout << ") fails: ";
		print_model_index(tmp);
		std::cout << std::endl;
		return 1;
	}

	return 0;
}
