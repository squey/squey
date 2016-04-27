/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVCharsetDetect.h>

PVRush::PVCharsetDetect::PVCharsetDetect() : nsUniversalDetector(), _found(false)
{
}

void PVRush::PVCharsetDetect::Report(const char* charset)
{
	_charset = charset;
	_found = true;
}

void PVRush::PVCharsetDetect::Reset()
{
	nsUniversalDetector::Reset();
	_charset.clear();
	_found = false;
}

std::string const& PVRush::PVCharsetDetect::GetCharset() const
{
	return _charset;
}

bool PVRush::PVCharsetDetect::found() const
{
	return _found;
}
