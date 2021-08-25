//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVDateTimeParser.h>

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QTextCodec>

#include <iostream>

#include <pvkernel/core/inendi_assert.h>

// Reference values are computed thanks to http://www.epochconverter.com/
int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " file" << std::endl;
		std::cerr << "where file is an UTF-8 encoded CSV-like file with :" << std::endl;
		std::cerr << "time string,time format,epoch in ms,hour in day,minute in day,seconds in day"
		          << std::endl
		          << std::endl;
		std::cerr << "For instance:" << std::endl;
		std::cerr << "01/01/1970 00:00:01,d/M/yyyy h:m:s,1" << std::endl;
		return 1;
	}

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
	size_t line_num = 0;
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

		PV_ASSERT_VALID(parser.mapping_time_to_cal(time_str, cal), "time_str",
		                qPrintable(time_str));

		// Copy the date time parser and try it !
		PVCore::PVDateTimeParser parser_copy(parser);

		PV_ASSERT_VALID(parser_copy.mapping_time_to_cal(time_str, cal_copy), "time_str",
		                qPrintable(time_str));

		UErrorCode err = U_ZERO_ERROR;
		PV_ASSERT_VALID(cal->equals(*cal_copy, err_) != 0, "line_num", line_num, "time_str",
		                qPrintable(time_str), "cal", cal->getTime(err), "cal_copy",
		                cal_copy->getTime(err));

		err = U_ZERO_ERROR;
		int64_t res = cal->getTime(err);
		int32_t s = cal->get(UCAL_SECOND, err);
		int32_t m = cal->get(UCAL_MINUTE, err);
		int32_t h = cal->get(UCAL_HOUR_OF_DAY, err);

		PV_VALID(res, res_ref);
		PV_VALID(h, h_ref);
		PV_VALID(m, m_ref);
		PV_VALID(s, s_ref);

		++line_num;
	}

	delete cal;

	return 0;
}
