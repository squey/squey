#ifndef PVINPUT_FILE_H
#define PVINPUT_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <boost/shared_ptr.hpp>
#include <QString>

namespace PVRush {

class LibKernelDecl PVInput {
public:
	typedef boost::shared_ptr<PVInput> p_type;
	typedef size_t input_offset;
public:
	PVInput();
	virtual ~PVInput();
public:
	// This method must read at most n bytes and put the result in buffer and returns the number of bytes actually read.
	// It returns 0 if no more data is available
	virtual size_t operator()(char* buffer, size_t n) = 0;
	virtual p_type clone() const = 0;
	// This method must return the current input offset of the object. For instance, for a file, it would be
	// the current offset of the file opened.
	virtual input_offset current_input_offset() = 0;
	// Seek to the beggining of the input
	virtual void seek_begin() = 0;
	virtual QString human_name() = 0;
};

class LibKernelDecl PVInputException {
public:
	virtual std::string const& what() const = 0;
};

typedef PVInput::p_type PVInput_p;

}

#define IMPL_INPUT(T) \
	T::p_type T::clone() const\
	{\
		return p_type(new T(*this));\
	}\

#define CLASS_INPUT(T) \
	public:\
		virtual p_type clone() const;\

#endif
