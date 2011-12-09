#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMatrix.h>
#include <QString>

int main()
{
	//PVCore::PVMatrix<int> a(10,10);
	//a.at(1,1) = 4;

	QString str("salut");
	PVCore::PVMatrix<QString> b;
	b.resize(10,10,str);

	return 0;
}
