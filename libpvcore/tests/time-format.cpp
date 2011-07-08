#include <pvcore/PVDateTimeParser.h>
#include <stdint.h>

#include <QCoreApplication>
#include <QString>
#include <QFile>
#include <QTextStream>
#include <QTextCodec>

#include <iostream>

// Reference values are computed thanks to http://www.epochconverter.com/
int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " file" << std::endl;
		std::cerr << "where file is an UTF-8 encoded CSV-like file with :" << std::endl;
		std::cerr << "time string,time format,epoch in ms" << std::endl << std::endl;
		std::cerr << "For instance:" << std::endl;
		std::cerr << "01/01/1970 00:00:01,d/M/yyyy h:m:s,1" << std::endl;
		return 1;
	}

	QCoreApplication app(argc,argv);


	QFile file(argv[1]);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		std::cerr << "Unable to open " << argv[1] << std::endl;
		return 1;
	}
	QTextStream in(&file);
	// Cf. http://www.iana.org/assignments/character-sets, UTF-8 has the MIB number 106.
	in.setCodec(QTextCodec::codecForMib(106));

	UErrorCode err_ = U_ZERO_ERROR;
	Calendar* cal = Calendar::createInstance(err_);
	while (!in.atEnd()) {
		QString line = in.readLine().trimmed();
		if (line.size() == 0) {
			continue;
		}
		QStringList params = line.split(QChar(','));
		QString const& time_str = params[0];
		QString const& time_format = params[1];
		bool ok = false;
		int64_t res_ref = params[2].toLongLong(&ok);

		PVCore::PVDateTimeParser parser(QStringList() << time_format);
		parser.mapping_time_to_cal(time_str, cal);
		UErrorCode err = U_ZERO_ERROR;
		int64_t res = cal->getTime(err);
		std::cout << qPrintable(time_str) << "," << qPrintable(time_format) << "," << res;
		if (res != res_ref) {
			std::cout << " failed (!= " << res_ref << ")" << std::endl;
			return 1;
		}
		std::cout << std::endl;
	}

	delete cal;

	return 0;
}
