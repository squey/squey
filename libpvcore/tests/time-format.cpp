#include <pvcore/PVDateTimeParser.h>
#include <QStringList>
#include <QLocale>
#include <QTextCodec>
#include <QCoreApplication>

#include <iostream>

int main(int argc, char** argv)
{
	const char* str_test[] = {"2000 Feb 1 1:0:0 AM +0200", "2000 Feb 1 1:0:0 PM +0200", "2000 Feb 1 +0000", "2000 Febrero 1", "2000 Févr. 1 +0200", "2000 Février 1", "2000 Feb 1", "2000 ФЕВРАЛЬ 1", "2000 شباط / فبراير 1"};
	QStringList formats = QStringList() << QString("yyyy MMM d h:m:m a") << QString("yyyy MMM d Z") << QString("yyyy MMM d");
	PVCore::PVDateTimeParser parser(formats);

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	UErrorCode err_ = U_ZERO_ERROR;
	Calendar* cal = Calendar::createInstance(err_);
	for (size_t i = 0; i < sizeof(str_test)/(sizeof(char*)); i++) {
		std::cout << str_test[i] << ": ";
		bool ret = parser.mapping_time_to_cal(QString::fromUtf8(str_test[i]), cal);
		if (!ret) {
			std::cout << "unable to map" << std::endl;
			continue;
		}
		UErrorCode err = U_ZERO_ERROR;
		std::cout << cal->getTime(err) << std::endl;
	}
	delete cal;

	return 0;
}
