#include <picviz/PVRoot.h>
#include <picviz/PVMappingFilter.h>
#include <pvrush/PVNraw.h>

#include <iostream>
#include <QCoreApplication>

#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " plugin file" << std::endl;
		return 1;
	}

	init_env();
	QCoreApplication app(argc, argv);

	Picviz::PVRoot root;

	Picviz::PVMappingFilter::p_type mapping_filter = LIB_FILTER(Picviz::PVMappingFilter)::get().get_filter_by_name(argv[1]);
	if (!mapping_filter) {
		std::cerr << "Filter " << argv[1] << " does not exist." << std::endl;
		return 1;
	}

	// Assume text file is in UTF8
	QFile f(argv[2]);
	if (!f.open(QFile::ReadOnly)) {
		std::cerr << "Unable to open " << argv[2] << std::endl;
		return 1;
	}
	QTextStream s(&f);
	s.setCodec("UTF-8");
	PVRush::PVNraw::nraw_table_line strings;
	while (true) {
		QString str = s.readLine();
		if (str.isNull()) {
			break;
		}
		strings.push_back(str);
	}

	float* mapped = (float*) malloc(strings.size()*sizeof(float));
	mapping_filter->set_dest_array(strings.size(), mapped);
	mapping_filter->operator()(strings);

	for (size_t i = 0; i < strings.size(); i++) {
		std::cout << "'" << qPrintable(strings[i]) << "','" << mapped[i] << "'" << std::endl;
	}

	return 0;
}
