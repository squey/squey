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
			size_t new_size = std::distance(begin, end) + _path_name.size();
			PVCore::PVElement* new_element = PVUnicodeSource<>::add_uninitialized_element(new_size);

			std::copy(_path_name.begin(), _path_name.end(), new_element->begin());
			*(new_element->begin() + _path_name.size()) =
			    PVRush::PVUnicodeSource<>::MULTI_INPUTS_SEPARATOR;
			std::copy(begin, end - 1, new_element->begin() + _path_name.size() + 1);
		} else {
			PVUnicodeSource<>::add_element(begin, end);
		}
	}

  private:
	std::string _path_name;
	PVFileDescription* _input_desc = nullptr;
};
} // namespace PVRush

#endif // __PVTEXTFILESOURCE_H__
