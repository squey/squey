
#include <pvkernel/core/PVFileHelper.h>
#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/PVLogger.h>

#include <iostream>
#include <stdio.h>

#include <assert.h>

#define FILENAME "/tmp/test_file_helper.test"

int main()
{
	std::cout << "testing for a non existing file (a warning must be printed)" << std::endl;
	ASSERT_VALID(PVCore::PVFileHelper::is_already_opened(FILENAME) == false);

	FILE* fp = fopen(FILENAME, "wt");

	std::cout << "testing for an already opened file" << std::endl;
	ASSERT_VALID(PVCore::PVFileHelper::is_already_opened(FILENAME) == true);

	char c;
	fwrite(&c, 1, 1, fp);

	fclose(fp);

	std::cout << "testing for an existing but not opened file" << std::endl;
	ASSERT_VALID(PVCore::PVFileHelper::is_already_opened(FILENAME) == false);

	(void)remove(FILENAME);

	return 0;
}
