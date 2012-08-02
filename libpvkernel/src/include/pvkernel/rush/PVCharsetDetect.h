/**
 * \file PVCharsetDetect.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCHARSETDETECT_FILE_H
#define PVCHARSETDETECT_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/uchardetect/nscore.h>
#include <pvkernel/rush/uchardetect/nsUniversalDetector.h>

#include <string>

namespace PVRush {

class LibKernelDecl PVCharsetDetect : public nsUniversalDetector {
public:
	PVCharsetDetect();

protected:
	virtual void Report(const char* charset);
	virtual void Reset();

public:
	std::string const& GetCharset() const;
	bool found() const;

protected:
	std::string _charset;
	bool _found;
};

}

#endif
