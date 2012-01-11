#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMatrix.h>
#include <QString>

#include <iostream>

int main()
{
	//PVCore::PVMatrix<int> a(10,10);
	//a.at(1,1) = 4;

	QString str("salut");
	PVCore::PVMatrix<QString> b;
	b.resize(10,10,str);
	b.resize_nrows(40, QString());
	b.resize_nrows(60, QString());

	// Iterators
	PVCore::PVMatrix<float> mf;
	mf.resize(10, 10);
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			float v = (float)i + ((float)j/10.0f);
			mf.at(i,j) = v;
		}
	}

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << mf.at(i,j) << "\t";
		}
		std::cout << std::endl;
	}

	PVCore::PVMatrix<float>::column col = mf.get_col(5);
	for (size_t i = 0; i < col.size(); i++) {
		std::cout << col.at(i) << std::endl;
	}

	return 0;
}
