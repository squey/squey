#include <pvkernel/core/PVDateTimeParser.h>
#include <pvkernel/core/stdint.h>

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
		std::cerr << "time string,time format,epoch in ms,hour in day,minute in day,seconds in day" << std::endl << std::endl;
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
	Calendar* cal_copy = Calendar::createInstance(err_);
	while (!in.atEnd()) {
		QString line = in.readLine().trimmed();
		if (line.size() == 0)
			continue;

		QStringList params = line.split(QChar(','));
		if (params.size() < 6)
			continue;
		QString const& time_str = params[0];
		QString const& time_format = params[1];
		bool ok = false;
		int64_t res_ref = params[2].toLongLong(&ok);
		if (!ok)
			continue;
		int32_t h_ref = params[3].toLongLong(&ok);
		if (!ok)
			continue;
		int32_t m_ref = params[4].toLongLong(&ok);
		if (!ok)
			continue;
		int32_t s_ref = params[5].toLongLong(&ok);
		if (!ok)
			continue;

		PVCore::PVDateTimeParser parser(QStringList() << time_format);
		if (!parser.mapping_time_to_cal(time_str, cal)) {
			std::cout << "Unable to parse " << qPrintable(time_str) << std::endl;
			return 1;
		}
		// Copy the date time parser and try it !
		PVCore::PVDateTimeParser parser_copy(parser);
		if (!parser_copy.mapping_time_to_cal(time_str, cal_copy)) {
			std::cout << "Unable to parse " << qPrintable(time_str) << " with a copy of the parser object." << std::endl;
			return 1;
		}
		if (!cal->equals(*cal_copy, err_)) {
			UErrorCode err = U_ZERO_ERROR;
			std::cout << "A copy of the parser object gave us another value: org " << cal->getTime(err) << " != " << cal_copy->getTime(err) << std::endl;
			return 1;
		}
		UErrorCode err = U_ZERO_ERROR;
		int64_t res = cal->getTime(err);
		int32_t s = cal->get(UCAL_SECOND, err);
		int32_t m = cal->get(UCAL_MINUTE, err);
		int32_t h = cal->get(UCAL_HOUR_OF_DAY, err);
		std::cout << qPrintable(time_str) << "," << qPrintable(time_format) << "," << res << "," << h_ref << "," << m_ref << "," << s_ref;
		if (res != res_ref) {
			std::cout << " failed (res!= " << res_ref << ")" << std::endl;
			return 1;
		}
		if (h != h_ref) {
			std::cout << " failed (h!=" << h << ")" << std::endl;
			return 1;
		}
		if (m != m_ref) {
			std::cout << " failed (m!=" << m << ")" << std::endl;
			return 1;
		}
		if (s != s_ref) {
			std::cout << " failed (s!=" << s << ")" << std::endl;
			return 1;
		}
		std::cout << std::endl;
	}

	delete cal;

	return 0;
}
