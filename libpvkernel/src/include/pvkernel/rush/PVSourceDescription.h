/**
 * \file PVSourceDescription.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVSOURCEDESCRIPTION_H_
#define PVSOURCEDESCRIPTION_H_

#include <pvkernel/core/PVSharedPointer.h>

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>

namespace PVRush
{

class PVSourceDescription
{
public:
	typedef typename PVCore::PVSharedPtr<PVSourceDescription> shared_pointer;

public:
	PVSourceDescription() :
		_inputs(),
		_source_creator_p(),
		_format()
	{
	}

	PVSourceDescription(
		const PVRush::PVInputType::list_inputs& inputs,
		PVRush::PVSourceCreator_p source_creator_p,
		const PVRush::PVFormat& format
	) :
		_inputs(inputs),
		_source_creator_p(source_creator_p),
		_format(format)
	{
	}

	bool operator==(const PVSourceDescription& other) const;
	bool operator!=(const PVSourceDescription& other) const { return !operator==(other); }

	void set_inputs(const PVRush::PVInputType::list_inputs inputs) { _inputs = inputs; }
	void set_source_creator(PVRush::PVSourceCreator_p source_creator_p) { _source_creator_p = source_creator_p; }
	void set_format(PVRush::PVFormat format) { _format = format; }

	const PVRush::PVInputType::list_inputs& get_inputs() const { return _inputs; }
	PVRush::PVSourceCreator_p get_source_creator() const { return _source_creator_p; }
	PVRush::PVFormat get_format() const { return _format; }

private:
	PVRush::PVInputType::list_inputs _inputs;
	PVRush::PVSourceCreator_p _source_creator_p;
	PVRush::PVFormat _format;
};

}

#endif /* PVSOURCEDESCRIPTION_H_ */
