#ifndef PVCORE_TYPEINFO_H
#define PVCORE_TYPEINFO_H

#include <typeinfo>

namespace PVCore {

/*! \brief Defines a copyable std::type_info.
 *
 * This is mainly inspired by
 * http://alfps.wordpress.com/2010/06/15/cppx-unique-identifier-values-via-stdtype_info/.
 * This allows to store various definition of a class type. When using this
 * class, one must be aware of the issue of using this accross libraries
 * boundaries under Windows. Indeed, two typeid can be equal but still refers
 * to different class types.
 */
class PVTypeInfo
{
private:
	std::type_info const*   pStdInfo_;

public:
	virtual ~PVTypeInfo() {}

	PVTypeInfo( std::type_info const& stdInfo )
		: pStdInfo_( &stdInfo )
	{}

	char const* name() const { return pStdInfo_->name(); }

	bool operator!=( PVTypeInfo const& other ) const
	{
		return !!(*pStdInfo_ != *other.pStdInfo_);      // "!!" for MSVC non-std ops.
	}

	bool operator<( PVTypeInfo const& other ) const
	{
		return !!pStdInfo_->before( *other.pStdInfo_ ); // "!!" for MSVC non-std ops.
	}

	bool operator<=( PVTypeInfo const& other ) const
	{
		return !other.pStdInfo_->before( *pStdInfo_ );
	}

	bool operator==( PVTypeInfo const& other ) const
	{
		return !!(*pStdInfo_ == *other.pStdInfo_);      // "!!" for MSVC non-std ops.
	}

	bool operator>=( PVTypeInfo const& other ) const
	{
		return !pStdInfo_->before( *other.pStdInfo_ );
	}

	bool operator>( PVTypeInfo const& other ) const
	{
		return !!other.pStdInfo_->before( *pStdInfo_ ); // "!!" for MSVC non-std ops.
	}
};

}

#endif
