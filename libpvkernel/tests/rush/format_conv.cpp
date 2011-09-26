#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVFormatVersion.h>

#include <QCoreApplication>
#include <QFile>
#include <QTextStream>
#include <QDomDocument>

#include <iostream>

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " format" << std::endl;
		return 1;
	}

	// Read a format, convert it to the last vversion and dump its XML content + its internal representation
	
	QFile f(argv[1]);
	if (!f.open(QIODevice::ReadOnly)) {
		std::cerr << "Unable to open " << argv[1] << std::endl;
		return 1;
	}
	QTextStream ft(&f);
	ft.setCodec("UTF-8");
	QDomDocument doc;
	doc.setContent(ft.readAll(), false);
	PVRush::PVFormatVersion::to_current(doc);

	std::cout << doc.toString().toUtf8().constData() << std::endl;

	return 0;
}
