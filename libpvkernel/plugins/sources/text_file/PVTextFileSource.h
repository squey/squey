/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */
#ifndef __PVTEXTFILESOURCE_H__
#define __PVTEXTFILESOURCE_H__

#include <iterator>
#include <fcntl.h>
#include <memory>

#include <QString>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVUnicodeSource.h>

namespace PVRush
{

class PVTextFileSource : public PVUnicodeSource<>
{
  public:
	PVTextFileSource(PVInput_p input,
	                 PVFileDescription* input_desc,
	                 size_t chunk_size,
	                 const alloc_chunk& alloc = alloc_chunk())
	    : PVUnicodeSource<>(input, chunk_size, alloc), _input_desc(input_desc)
	{
		_path_name = QFileInfo(input_desc->path()).fileName().toStdString();
	}

  public:
	void add_element(char* begin, char* end) override
	{
		if (_input_desc->multi_inputs()) {
			size_t element_size = std::distance(begin, end);

			_temp_elements.emplace_back();
			std::string& new_element = _temp_elements.back();
			new_element.resize(element_size + _path_name.size() + 1);

			size_t len = 0;
			std::copy(_path_name.begin(), _path_name.end(), new_element.begin());
			new_element[_path_name.size()] = PVRush::PVUnicodeSource<>::MULTI_INPUTS_SEPARATOR;
			len = _path_name.size() + 1;
			std::copy(begin, end, new_element.begin() + len);
			char* new_begin = const_cast<char*>(new_element.data());
			char* new_end = const_cast<char*>(new_element.data() + new_element.size());
			PVUnicodeSource<>::add_element(new_begin, new_end);
		} else {
			PVUnicodeSource<>::add_element(begin, end);
		}
	}

  private:
	std::string _path_name;
	std::deque<std::string> _temp_elements; // FIXME : should not increase memory consumption
	PVFileDescription* _input_desc = nullptr;
};
} // namespace PVRush

#endif // __PVTEXTFILESOURCE_H__
