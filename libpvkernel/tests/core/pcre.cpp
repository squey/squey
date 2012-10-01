#include <pcrecpp.h>
#include <iostream>

int main()
{
	pcrecpp::RE re("héllo", PCRE_CASELESS|PCRE_UTF8);
	std::cout << re.FullMatch("hÉllo") << std::endl;
	std::cout << re.FullMatch("hello") << std::endl;
	std::cout << re.FullMatch("ahélloz") << std::endl;
	std::cout << re.PartialMatch("ahélloz") << std::endl;

	pcrecpp::RE my_re = re;
	std::cout << my_re.FullMatch("hÉllo") << std::endl;
	std::cout << my_re.FullMatch("hello") << std::endl;
	std::cout << my_re.FullMatch("ahélloz") << std::endl;
	std::cout << my_re.PartialMatch("ahélloz") << std::endl;
	return 0;
}
