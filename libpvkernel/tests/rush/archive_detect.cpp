#include "../plugins/input_types/file/extract.h"
#include <iostream>

#include <QCoreApplication>

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " file" << std::endl;
		return 1;
	}

	QCoreApplication app(argc, argv); // So that QString uses the good locale for its converion
	QString file = QString::fromLocal8Bit(argv[1]);
	bool ret = is_archive(file);
	if (ret) {
		std::cout << file.toUtf8().constData() << " is an archive." << std::endl;
	}
	else {
		std::cout << file.toUtf8().constData() << " isn't an archive." << std::endl;
	}
	return 0;
}
