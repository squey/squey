#ifndef PVKERNEL_CORE_PVSERIALIZEDSOURCE_HPP
#define PVKERNEL_CORE_PVSERIALIZEDSOURCE_HPP

#include <vector>
#include <string>

// TODO : input_desc is "designed" to be use for files. To make it works with others inputs, we
// should make PVSerializedSource a template class with the description type as template argument or
// create smart parser on this string.

namespace PVCore
{
struct PVSerializedSource {
	std::vector<std::string> input_desc;
	std::string sc_name;
	std::string format_name;
	std::string format_path;

	bool need_credential() const { return sc_name != "text_file"; }
};
} // namespace PVCore

#endif
