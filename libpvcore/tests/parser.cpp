#include <pvcore/PVXmlParamParser.h>
#include <iostream>

int main(int argc, char** argv)
{
	if (argc < 1) {
		std::cerr << "Usage: " << argv[0] << "file_format" << std::endl;
		return 1;
	}
	PVCore::PVXmlParamParser parser(argv[1]);

	return 0;
}
