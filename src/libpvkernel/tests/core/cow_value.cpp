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

#include <pvkernel/core/PVCowValue.h>
#include <pvkernel/core/squey_assert.h>

#include <vector>

/**
 * Counts its own copies, so that a test can assert on what was duplicated
 * rather than on what was merely reachable.
 */
struct counted {
	static size_t copies;

	explicit counted(int v = 0) : value(v) {}
	counted(counted const& o) : value(o.value) { ++copies; }
	counted& operator=(counted const&) = delete;

	int value;
};

size_t counted::copies = 0;

using holder = PVCore::PVCowValue<counted>;

/**
 * Reading and writing without a snapshot around never duplicates anything.
 */
static void writing_alone_costs_no_copy()
{
	counted::copies = 0;

	holder h(1);
	PV_ASSERT_VALID(h.read().value == 1);

	h.write().value = 2;
	h.write().value = 3;

	PV_VALID(h.read().value, 3);
	PV_VALID(counted::copies, size_t(0));
	PV_ASSERT_VALID(not h.is_shared());
}

/**
 * A snapshot keeps what it captured, and pays for exactly one copy however many
 * writes follow it.
 */
static void a_snapshot_keeps_what_it_captured()
{
	counted::copies = 0;

	holder h(1);
	auto const s = h.snapshot();
	PV_ASSERT_VALID(h.is_shared());
	PV_VALID(counted::copies, size_t(0), "why", "taking a snapshot must not copy");

	h.write().value = 2;
	PV_VALID(counted::copies, size_t(1), "why", "the first write under a snapshot detaches");
	PV_VALID(s->value, 1, "why", "the snapshot must not see the write");
	PV_VALID(h.read().value, 2);
	PV_ASSERT_VALID(not h.is_shared());

	h.write().value = 3;
	PV_VALID(counted::copies, size_t(1), "why", "further writes are already detached");
	PV_VALID(s->value, 1);
	PV_VALID(h.read().value, 3);
}

/**
 * What restoring is for: going back to a captured value, then moving on from it
 * without damaging what is still captured.
 */
static void restoring_goes_back_without_damaging_the_snapshot()
{
	holder h(1);
	auto const first = h.snapshot();

	h.write().value = 2;
	auto const second = h.snapshot();

	h.restore(first);
	PV_VALID(h.read().value, 1);
	PV_ASSERT_VALID(h.is(first), "why", "restoring must land on the very captured value");

	// Moving on from a restored value has to leave both captures alone: this is
	// the redo branch being dropped, not the history being rewritten.
	h.write().value = 4;
	PV_VALID(first->value, 1);
	PV_VALID(second->value, 2);
	PV_VALID(h.read().value, 4);

	// Landing on the same step twice must be as good as landing on it once.
	h.restore(first);
	h.restore(first);
	PV_VALID(h.read().value, 1);
	PV_VALID(first->value, 1);
}

/**
 * Copying a holder shares rather than duplicates, and whichever copy writes
 * first is the one that pays.
 */
static void copying_a_holder_shares_the_value()
{
	counted::copies = 0;

	holder a(1);
	holder b(a);
	PV_VALID(counted::copies, size_t(0));
	PV_ASSERT_VALID(a.is_shared() and b.is_shared());

	b.write().value = 2;
	PV_VALID(counted::copies, size_t(1));
	PV_VALID(a.read().value, 1);
	PV_VALID(b.read().value, 2);
	PV_ASSERT_VALID(not a.is_shared() and not b.is_shared());
}

/**
 * The property the whole undo design rests on: capturing a set of values and
 * changing one of them duplicates that one alone. Were this to copy the lot,
 * a history of layer stacks would cost a layer stack per step.
 */
static void a_step_duplicates_only_what_it_changes()
{
	constexpr size_t count = 100;

	std::vector<holder> values;
	values.reserve(count);
	for (size_t i = 0; i < count; ++i) {
		values.emplace_back(int(i));
	}

	std::vector<holder::snapshot_type> step;
	step.reserve(count);
	for (auto const& v : values) {
		step.push_back(v.snapshot());
	}

	counted::copies = 0;
	values[7].write().value = -1;
	PV_VALID(counted::copies, size_t(1), "why", "one changed value must cost one copy");

	for (size_t i = 0; i < count; ++i) {
		if (i == 7) {
			PV_ASSERT_VALID(not values[i].is(step[i]));
			PV_VALID(step[i]->value, 7, "why", "the captured value must be the old one");
			PV_VALID(values[i].read().value, -1);
		} else {
			PV_ASSERT_VALID(values[i].is(step[i]), "why", "untouched values must stay shared");
			PV_VALID(values[i].read().value, int(i));
		}
	}
}

int main()
{
	writing_alone_costs_no_copy();
	a_snapshot_keeps_what_it_captured();
	restoring_goes_back_without_damaging_the_snapshot();
	copying_a_holder_shares_the_value();
	a_step_duplicates_only_what_it_changes();

	return 0;
}
