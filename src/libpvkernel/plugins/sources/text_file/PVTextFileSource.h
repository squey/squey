/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVTEXTFILESOURCE_H__
#define __PVTEXTFILESOURCE_H__

#include <iterator>
#include <fcntl.h>
#include <memory>

#include <QString>

#include <pvkernel/core/PVTextChunk.h>
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
			constexpr const size_t separator_size = 1;
			bool trim_cr = (*(end - 1) == 0xd); // @note Remove \r for \r\n new-line (windows style)
			size_t new_size =
			    _path_name.size() + separator_size + std::distance(begin, end) - trim_cr;
			PVCore::PVElement* new_element = PVUnicodeSource<>::add_uninitialized_element(new_size);
			std::copy(_path_name.begin(), _path_name.end(), new_element->begin());
			*(new_element->begin() + _path_name.size()) =
			    PVRush::PVUnicodeSource<>::MULTI_INPUTS_SEPARATOR;
			std::copy(begin, end - trim_cr,
			          new_element->begin() + _path_name.size() + separator_size);
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
