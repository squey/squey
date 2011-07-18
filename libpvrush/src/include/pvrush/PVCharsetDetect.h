#ifndef PVCHARSETDETECT_FILE_H
#define PVCHARSETDETECT_FILE_H

#include <pvcore/general.h>
#include <pvrush/uchardetect/nscore.h>
#include <pvrush/uchardetect/nsUniversalDetector.h>

#include <string>

namespace PVRush {

class LibRushDecl PVCharsetDetect : public nsUniversalDetector {
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
