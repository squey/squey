/* * MIT License
 *
 * © Squey, 2026
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

#ifndef PVCORE_PVCOWVALUE_H
#define PVCORE_PVCOWVALUE_H

#include <memory>
#include <type_traits>
#include <utility>

namespace PVCore
{

/**
 * \class PVCowValue
 *
 * Holds a value that snapshots can be taken of without copying it.
 *
 * A snapshot is a shared pointer to the value as it stands, so taking one costs
 * a refcount increment whatever the value weighs. The copy happens later and
 * only if it has to: the next write to a value some snapshot still holds
 * detaches it first, leaving the snapshot with the version it captured.
 *
 * That is what makes an undo history affordable here. A step that only changes
 * the selection duplicates the selection alone; the layers it did not touch
 * stay shared between every step that did not touch them either. A history of
 * unchanged values costs nothing but pointers.
 *
 * The read()/write() split is what tells a mutation from a mere look. Calling
 * write() without writing is harmless -- at worst it buys one copy that nothing
 * needed -- whereas reaching a mutation through read() would let a snapshot see
 * a value change under it, so the accessors of a holder should hand out write()
 * from their non-const overload and read() from their const one, and let
 * const-correctness sort the call sites out.
 *
 * Not thread safe: the use_count() test in write() is a read followed by a
 * decision, so two threads mutating the same holder can both conclude they own
 * the value alone. Holders are meant to be mutated by one thread at a time,
 * which is what the surrounding code already assumes.
 */
template <typename T>
class PVCowValue
{
  public:
	using value_type = T;
	using snapshot_type = std::shared_ptr<const T>;

  public:
	/**
	 * Builds the held value in place, forwarding to one of its constructors.
	 *
	 * A single argument that is itself a holder is left to the copy and move
	 * constructors below, which this one would otherwise outrank when handed a
	 * non-const holder.
	 */
	template <typename... Args>
	    requires(sizeof...(Args) != 1 ||
	             not(std::is_same_v<std::remove_cvref_t<Args>, PVCowValue> && ...))
	explicit PVCowValue(Args&&... args)
	    : _value(std::make_shared<T>(std::forward<Args>(args)...))
	{
	}

	/**
	 * Copying a holder shares the value rather than duplicating it; whichever
	 * of the two writes first gets its own copy.
	 */
	PVCowValue(PVCowValue const&) = default;
	PVCowValue(PVCowValue&&) = default;
	PVCowValue& operator=(PVCowValue const&) = default;
	PVCowValue& operator=(PVCowValue&&) = default;

  public:
	/**
	 * The value, for reading. Never copies, never detaches.
	 */
	const T& read() const { return *_value; }

	/**
	 * The value, for writing. Detaches from any snapshot still holding it, so
	 * that what the snapshot captured cannot change afterwards.
	 */
	T& write()
	{
		detach();
		return *_value;
	}

	/**
	 * Replaces the whole value, which write() cannot do without cost: writing
	 * over a shared value would first duplicate the very contents about to be
	 * overwritten.
	 */
	void assign(T const& value)
	{
		if (_value.use_count() > 1) {
			_value = std::make_shared<T>(value);
		} else {
			*_value = value;
		}
	}

	void assign(T&& value)
	{
		if (_value.use_count() > 1) {
			_value = std::make_shared<T>(std::move(value));
		} else {
			*_value = std::move(value);
		}
	}

	/**
	 * Captures the value as it stands, in constant time.
	 */
	snapshot_type snapshot() const { return _value; }

	/**
	 * Puts back a value a snapshot captured. The holder then shares it with
	 * that snapshot, so the next write() detaches and the snapshot keeps what
	 * it captured -- restoring the same snapshot twice is therefore fine.
	 */
	void restore(snapshot_type s)
	{
		// The pointee was never really const: snapshot_type only spells out
		// that a snapshot is not a way in to mutate what it captured.
		_value = std::const_pointer_cast<T>(std::move(s));
	}

	/**
	 * Whether the value is the very one this snapshot captured, which is how a
	 * caller can tell an untouched value from one written back to the same
	 * contents.
	 */
	bool is(snapshot_type const& s) const { return _value == s; }

	/**
	 * Whether a write() would copy, i.e. whether anything else still holds the
	 * value. Meant for tests and for measuring, not for deciding anything.
	 */
	bool is_shared() const { return _value.use_count() > 1; }

  private:
	void detach()
	{
		if (_value.use_count() > 1) {
			_value = std::make_shared<T>(*_value);
		}
	}

  private:
	std::shared_ptr<T> _value;
};
} // namespace PVCore

#endif /* PVCORE_PVCOWVALUE_H */
