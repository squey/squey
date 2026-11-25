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

#include <pvkernel/core/PVFileHelper.h>
#include <pvkernel/core/squey_assert.h>

#include <iostream>
#include <string>
#include <cstdio>

#include <pvlogger.h>

#define FILENAME                                                                                   \
	"test_file_helper.test" // won't work on tmpfs, see https://lkml.org/lkml/2008/4/29/602

static void expect_suffixes(const char* path, const char* expected)
{
	PV_VALID(PVCore::PVFileHelper::suffixes_without_rotation(path).toStdString(),
	         std::string(expected), "path", std::string(path));
}

int main()
{
	// The numbers and the dates that logrotate and Suricata append, before or after a compression.
	expect_suffixes("/var/log/suricata/eve.json.1", "json");
	expect_suffixes("/var/log/suricata/eve.json.2.gz", "json.gz");
	expect_suffixes("/var/log/suricata/eve.json.gz.3", "json.gz");
	expect_suffixes("/var/log/suricata/eve.json-20261119", "json");
	expect_suffixes("/var/log/suricata/eve.json-2026-11-19_12.zst", "json.zst");
	expect_suffixes("/var/log/suricata/eve.json.1700000000", "json");
	// Names with dots, dashes and cases of their own.
	expect_suffixes("/tmp/report.2026.01.csv", "csv");
	expect_suffixes("/tmp/Traffic.CSV.GZ", "csv.gz");
	expect_suffixes("/tmp/notes.v2-final.txt", "v2-final.txt");
	expect_suffixes("/tmp/archive.d/README", "");
	expect_suffixes("/tmp/backup.1", "");

	std::cout << "testing for a non existing file (a warning must be printed)" << std::endl;
	PV_VALID(PVCore::PVFileHelper::is_already_opened(FILENAME), false);

	FILE* fp = fopen(FILENAME, "wt");
	pvlogger::info() << "fp=" << fp << std::endl;

	std::cout << "testing for an already opened file" << std::endl;
	PV_VALID(PVCore::PVFileHelper::is_already_opened(FILENAME), true);

	char c = 'T';
	fwrite(&c, 1, 1, fp);

	fclose(fp);

	std::cout << "testing for an existing but not opened file" << std::endl;
	PV_VALID(PVCore::PVFileHelper::is_already_opened(FILENAME), false);

	(void)remove(FILENAME);

	return 0;
}
