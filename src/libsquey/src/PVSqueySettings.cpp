//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVSqueySettings.h>

#include <squey/PVAnalysisHistory.h>

const PVCore::PVSetting<int> Squey::settings::undo_steps(
    {.key = "general.undo_steps",
     .type = PVCore::PVSettingType::Int,
     .default_value = int(PVAnalysisHistory::default_max_steps),
     .title = "Undo steps",
     .description = "Changes of selection that undo can go back through, in each "
                    "investigation opened afterwards. Each step takes the memory of a "
                    "selection.",
     .minimum = 1,
     .maximum = 1000,
     .keywords = {"history", "redo", "memory"}});

const PVCore::PVSetting<int> Squey::settings::sql_memory_percent(
    {.key = "sql.memory_percent",
     .type = PVCore::PVSettingType::Int,
     .default_value = 50,
     .title = "Memory of the SQL engine",
     .description = "Share of the memory of the machine that the SQL engine of a view may use. "
                    "Each view and each SQL console has an engine of its own, which reads this "
                    "setting as it starts. A query that needs more fails.",
     .minimum = 1,
     .maximum = 100,
     .unit = "%",
     .keywords = {"DuckDB", "memory_limit"}});

const PVCore::PVSetting<int> Squey::settings::sql_spill_percent(
    {.key = "sql.spill_percent",
     .type = PVCore::PVSettingType::Int,
     .default_value = 50,
     .title = "Disk space of the SQL engine",
     .description = "Share of the free space of the working directory that the SQL engine of a "
                    "view may fill with temporary files, for the queries that do not fit in "
                    "memory.",
     .minimum = 1,
     .maximum = 100,
     .unit = "%",
     .keywords = {"DuckDB", "temporary", "spill", "max_temp_percent"}});

const PVCore::PVSetting<int> Squey::settings::sql_threads(
    {.key = "sql.threads",
     .type = PVCore::PVSettingType::Int,
     .default_value = 0,
     .title = "Threads of the SQL engine",
     .description = "Threads the SQL engine of a view may run a query on, read as the engine "
                    "starts. 0 runs as many as the processor has.",
     .minimum = 0,
     .maximum = 1024,
     .keywords = {"DuckDB", "CPU", "processor", "cores"}});
