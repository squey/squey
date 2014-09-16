
#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVDataTreeObject.h>

#include <memory>

class Test;
typedef PVCore::PVSharedPtr<Test> Test_p;
typedef std::shared_ptr<Test> Test_p2;

struct A { };

typedef typename PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<A>, Test> data_tree_plotted_t ;

struct B: public data_tree_plotted_t
{ };

// FIXME: what is this program supposed to test?
int main()
{
	Test_p t1,t2;
	t1 = t2;

	Test_p t4(t1);
	t4.reset();

	return 0;
}
