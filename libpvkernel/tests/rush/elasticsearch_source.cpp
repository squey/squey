//
// MIT License
//
// © ESI Group, 2015
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

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVUtils.h>

// FIXME : It should not be include this way if plugins provide correct API.
#include "../../plugins/common/elasticsearch/PVElasticsearchInfos.h"
#include "../../plugins/common/elasticsearch/PVElasticsearchQuery.h"

#include <fstream>
#include "helpers.h"
#include "common.h"

#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/sources/elasticsearch.out";
#endif

int main()
{
	// FIXME it is a manual creation for InputType as API is not provided by input_type plugin.
	pvtest::init_ctxt();

	/*
	 * Set Up an ElasticSearchInfo.
	 * It contains all information required to connect with the server
	 */
	PVRush::PVElasticsearchInfos infos;
	infos.set_host("http://connectors.srv.picviz");
	infos.set_port(9200);
	infos.set_index("proxy_sample_geoip");
	infos.set_login("elastic");
	infos.set_password("changeme");
	infos.set_filter_path("category,http_method,login,mime_type,src_ip,status_"
	                      "code,time,time_spent,total_bytes,url,user_agent");

	/*
	 * Set Up an ElasticSearchQuery.
	 * It contains all information required to define data to extract
	 */
	std::string query_type = "json";
	std::string query_str = R"###({  
		"query":{  
			"constant_score":{  
				"filter":{  
					"bool":{  
						"must":[  
							{  
								"term":{  
									"http_method":"get"
								}
							},
							{  
								"bool":{  
									"should":[  
										{  
											"range":{  
												"time_spent":{  
													"gt":10000
												}
											}
										}
									],
									"must_not":[  
										{  
											"term":{  
												"mime_type":"image/jpeg"
											}
										}
									]
								}
							}
						],
						"must_not":[  
							{  
								"term":{  
									"login":"11437"
								}
							},
							{  
								"term":{  
									"login":"10715"
								}
							}
						]
					}
				}
			}
		}
	})###";
	PVRush::PVInputDescription_p ind(
	    new PVRush::PVElasticsearchQuery(infos, query_str.c_str(), query_type.c_str()));

	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("elasticsearch");
	PVRush::PVSourceCreator::source_p src = sc->create_source_from_input(ind);
	auto& source = *src;

	static constexpr const size_t MEGA = 1024 * 1024;
	PV_VALID(source.get_size(), MEGA * 219UL);

	std::string output_file = pvtest::get_tmp_filename();
	std::string ref_file_sorted = output_file + "_sorted";
	// Extract source and split fields.
	{
		std::ofstream ofs(output_file);

		std::chrono::duration<double> dur(0.);
		auto start = std::chrono::steady_clock::now();
		while (auto* pc = dynamic_cast<PVCore::PVTextChunk*>(source())) {
			auto end = std::chrono::steady_clock::now();
			dur += end - start;
			dump_chunk_csv(*pc, ofs);
			pc->free();
			start = std::chrono::steady_clock::now();
		}
		std::cout << dur.count();
	}

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	PVRush::PVUtils::sort_file(output_file.c_str());
	PVRush::PVUtils::sort_file(ref_file, ref_file_sorted.c_str());
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file_sorted));
#endif

	std::remove(output_file.c_str());
	std::remove(ref_file_sorted.c_str());

	return 0;
}
