#include <pvkernel/core/PVDateTimeParser.h>
#include <pvkernel/core/stdint.h>

#include <tbb/tick_count.h>

#include <QCoreApplication>
#include <QString>

#include <iostream>

int main(int argc, char** argv)
{
	QCoreApplication app(argc, argv);

	PVCore::PVDateTimeParser parser(QStringList() << QString("dd/MMM/yyyy"));
	UErrorCode err = U_ZERO_ERROR;
	Calendar* cal = Calendar::createInstance(err);

	QString time("01/Jan/2000");
	if (!parser.mapping_time_to_cal(time, cal)) {
		std::cerr << "Go and fix your time string, it can't be parsed !" << std::endl;
		return 1;
	}

	tbb::tick_count start,end;
	start = tbb::tick_count::now();
	int i;
	for (i = 0; i < 1000000; i++) {
		parser.mapping_time_to_cal(time, cal);
	}
	end = tbb::tick_count::now();
	std::cout << "Mapping of " << i << " time strings in " << (end-start).seconds() << "s" << std::endl;

	return 0;
}
