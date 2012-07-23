/**
 * \file PVSerializeArchiveFixError.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVSERIALIZEARCHIVEFIXERROR
#define PVCORE_PVSERIALIZEARCHIVEFIXERROR

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVTypeTraits.h>

#include <boost/shared_ptr.hpp>

namespace PVCore {

class PVSerializeObject;
class PVSerializeArchiveError;

class LibKernelDecl PVSerializeArchiveFixError
{
public:
	PVSerializeArchiveFixError(PVSerializeObject& so, boost::shared_ptr<PVSerializeArchiveError> const& ar_err):
		_so(so),
		_ar_err(ar_err)
	{ }
public:
	void error_fixed();
	PVSerializeArchiveError const& exception() const { return *_ar_err; }
	
	template <class T>
	bool exception_of_type() { return dynamic_cast<typename PVTypeTraits::pointer<T>::type>(_ar_err.get()) != NULL; }

	template <class T>
	T* exception_as() { return dynamic_cast<typename PVTypeTraits::pointer<T>::type>(_ar_err.get()); }
protected:
	PVSerializeObject& _so;
	boost::shared_ptr<PVSerializeArchiveError> _ar_err;
};

class LibKernelDecl PVSerializeArchiveFixAttribute: public PVSerializeArchiveFixError
{
public:
	PVSerializeArchiveFixAttribute(PVSerializeObject& so, boost::shared_ptr<PVSerializeArchiveError> const& ar_err, QString const& attribute):
		PVSerializeArchiveFixError(so, ar_err),
		_attribute(attribute)
	{ }

public:
	void fix(QVariant const& v);

private:
	QString _attribute;
};

typedef boost::shared_ptr<PVSerializeArchiveFixError> PVSerializeArchiveFixError_p;

}

#endif
