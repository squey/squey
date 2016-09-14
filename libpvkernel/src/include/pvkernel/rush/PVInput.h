/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVINPUT_FILE_H
#define PVINPUT_FILE_H

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <QString>

#include <pvkernel/rush/PVInput_types.h>

namespace PVRush
{

class PVInput
{
  public:
	typedef PVInput_p p_type;

  public:
	virtual ~PVInput() = default;

  public:
	// This method must read at most n bytes and put the result in buffer and returns the number of
	// bytes actually read.
	// It returns 0 if no more data is available
	virtual size_t operator()(char* buffer, size_t n) = 0;
	// Seek to the beggining of the input
	virtual void seek_begin() = 0;
	virtual QString human_name() = 0;
};

class PVInputException : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};
} // namespace PVRush

#define IMPL_INPUT(T)
#define CLASS_INPUT(T)

#endif
