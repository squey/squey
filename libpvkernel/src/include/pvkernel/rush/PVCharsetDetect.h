/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCHARSETDETECT_FILE_H
#define PVCHARSETDETECT_FILE_H

#include "nscore.h"
#include "nsUniversalDetector.h"

#include <string>

namespace PVRush
{

class PVCharsetDetect : public nsUniversalDetector
{
  public:
	PVCharsetDetect();

  protected:
	void Report(const char* charset) override;
	void Reset() override;

  public:
	std::string const& GetCharset() const;
	bool found() const;

  protected:
	std::string _charset;
	bool _found;
};
} // namespace PVRush

#endif
