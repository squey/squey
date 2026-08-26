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

// Binds a PVDuckDBQuery to a source or to a view: the column names it reads,
// and the rows each scope stands for.
//
// This lives apart from PVDuckDBQuery.cpp because that file is built as C++17
// (DuckDB's profiling_utils.hpp does not compile as C++23 under clang), while
// the Squey headers reaching the format require C++20 or later --
// PVClassLibrary.h uses `concept`. Keeping the two apart lets the column names
// and the layer stack be read without the DuckDB code ever seeing those
// headers.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVLayer.h>
#include <squey/PVLayerStack.h>
#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/rush/PVAxisFormat.h>
#include <pvkernel/rush/PVFormat.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace
{

/**
 * The SQL name of a column: the axis name, read from the format on each bind
 * rather than captured once, so renaming an axis or adding a column shows up
 * without rebuilding the query object. The source outliving the query is part
 * of the constructor's contract.
 */
std::function<std::string(size_t)> names_of(const Squey::PVSource& source)
{
	return [&source](size_t col) -> std::string {
		const QList<PVRush::PVAxisFormat>& axes = source.get_format().get_axes();
		if (col >= size_t(axes.size())) {
			return {}; // e.g. a column appended at runtime: named by position
		}
		return axes[int(col)].get_name().toStdString();
	};
}

/**
 * The scopes a view defines, resolved through @a current so that a query object
 * built before a view existed still finds one later.
 *
 * Each returns a pointer into the view, which owns what it hands back; a query
 * runs while the view is held still, which is the same contract the constructors
 * already carry.
 */
Squey::PVDuckDBQuery::Scopes scopes_of(std::function<const Squey::PVView*()> current)
{
	Squey::PVDuckDBQuery::Scopes scopes;

	scopes.selection = [current]() -> const PVCore::PVSelBitField* {
		const Squey::PVView* view = current();
		return view != nullptr ? &view->get_real_output_selection() : nullptr;
	};

	scopes.layers = [current]() -> const PVCore::PVSelBitField* {
		const Squey::PVView* view = current();
		return view != nullptr ? &view->get_layer_stack_output_layer().get_selection() : nullptr;
	};

	scopes.layer = [current](const std::string& name) -> const PVCore::PVSelBitField* {
		const Squey::PVView* view = current();
		if (view == nullptr) {
			return nullptr;
		}
		const Squey::PVLayerStack& stack = view->get_layer_stack();
		const QString wanted = QString::fromStdString(name);
		for (int i = 0; i < stack.get_layer_count(); ++i) {
			if (stack.get_layer_n(i).get_name() == wanted) {
				return &stack.get_layer_n(i).get_selection();
			}
		}
		return nullptr;
	};

	scopes.layer_names = [current]() -> std::vector<std::string> {
		std::vector<std::string> names;
		const Squey::PVView* view = current();
		if (view == nullptr) {
			return names;
		}
		const Squey::PVLayerStack& stack = view->get_layer_stack();
		for (int i = 0; i < stack.get_layer_count(); ++i) {
			names.emplace_back(stack.get_layer_n(i).get_name().toStdString());
		}
		return names;
	};

	return scopes;
}

/**
 * Every source of the project, the one @a primary reads first.
 *
 * A query can then join across sources, which is what a Python script reaching
 * for squey.source() has always been able to do. Order matters twice over: the
 * first entry is what a bare scope reads, and the rest keep their registration
 * order, which is the order a position argument counts namesakes in.
 *
 * The sisters are reached through the root rather than through the scene, to
 * match what the Python API exposes -- it walks the root's sources too.
 *
 * Each carries the scopes of its own current view: a joined source is read
 * through whatever view is active on it, not through the one asking.
 */
std::vector<Squey::PVDuckDBQuery::Source> project_sources(const Squey::PVSource& primary,
                                                          Squey::PVDuckDBQuery::Scopes primary_scopes)
{
	std::vector<Squey::PVDuckDBQuery::Source> sources;

	auto describe = [](const Squey::PVSource& source, Squey::PVDuckDBQuery::Scopes scopes) {
		Squey::PVDuckDBQuery::Source described;
		described.nraw = &source.get_rushnraw();
		described.name_of = names_of(source);
		described.scopes = std::move(scopes);
		described.name = source.get_name();
		return described;
	};

	sources.emplace_back(describe(primary, std::move(primary_scopes)));

	const Squey::PVRoot& root = primary.get_parent<Squey::PVScene>().get_parent<Squey::PVRoot>();
	for (const Squey::PVSource* sister : root.get_children<Squey::PVSource>()) {
		if (sister == nullptr || sister == &primary) {
			continue;
		}
		sources.emplace_back(
		    describe(*sister, scopes_of([sister]() { return sister->current_view(); })));
	}

	// The position that tells namesakes apart, filled in registration order so
	// that it agrees with what resolve_source() counts and what the "sources"
	// listing shows.
	for (size_t i = 0; i < sources.size(); ++i) {
		size_t position = 0;
		for (size_t j = 0; j < i; ++j) {
			position += size_t(sources[j].name == sources[i].name);
		}
		sources[i].position = position;
	}

	return sources;
}

} // namespace

Squey::PVDuckDBQuery::PVDuckDBQuery(const Squey::PVView& view)
    : PVDuckDBQuery(project_sources(view.get_parent<Squey::PVSource>(),
                                    scopes_of([&view]() { return &view; })))
{
}

Squey::PVDuckDBQuery::PVDuckDBQuery(const Squey::PVSource& source)
    // Read per query rather than at construction: a source has no view until one
    // is created, and which one is active can change under a long-lived query
    // object.
    : PVDuckDBQuery(
          project_sources(source, scopes_of([&source]() { return source.current_view(); })))
{
}
