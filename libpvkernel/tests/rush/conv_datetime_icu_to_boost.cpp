/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/rush/PVFormat.h>

#include <pvcop/pvcop.h>
#include <pvcop/types/datetime_us.h>

#include <iostream>


int main()
{
	std::vector<std::pair<std::string, std::string>> datetime_array;

	//list of couple ("datetime","icu format") to test
	datetime_array.push_back(std::make_pair("1223884800", "epoch"));
	datetime_array.push_back(std::make_pair("1334036784.745", "epoch"));
	datetime_array.push_back(std::make_pair("Mon Oct 13 10:00:00 GMT 2008", "eee MMM d H:m:ss V yyyy"));
	datetime_array.push_back(std::make_pair("2017-03-19 10:00:59.001", "yyyy-M-d H:m:ss.S"));
	datetime_array.push_back(std::make_pair("2012-03-19T10:00:59.000000000Z", "yyyy-M-d'T'H:m:ss.S"));
	datetime_array.push_back(std::make_pair("2015-03-27T00:00:07.1882+01:00", "yyyy-M-d'T'H:m:s.SZ"));
	datetime_array.push_back(std::make_pair("2015-03-26 23:42:35", "yyyy-M-d H:m:s"));
	datetime_array.push_back(std::make_pair("1996.07.10 AD at 15:08:56 PDT", "yyyy.MM.dd G 'at' HH:mm:ss zzz"));
	datetime_array.push_back(std::make_pair("Wed, July 10, '96", "EEE, MMM d, ''yy"));
	datetime_array.push_back(std::make_pair("12:08 PM", "h:mm a"));
	datetime_array.push_back(std::make_pair("12 o'clock PM, Pacific Daylight Time", "hh 'o''clock' a, zzzz"));
	datetime_array.push_back(std::make_pair("0:00 PM, PST", "K:mm a, z"));
	datetime_array.push_back(std::make_pair("01996.July.10 AD 12:08 PM", "yyyyy.MMMM.dd GGG hh:mm aaa"));

	// db::array table of datetime_us type
	pvcop::types::formatter_datetime_us dtf_us("");
	pvcop::db::array out_array =
			    pvcop::db::array(dtf_us.storage_type_id(), datetime_array.size());

	for (size_t i = 0; i < datetime_array.size(); i++) {
		std::string boost_datetime_format = PVRush::PVFormat::convert_ICU_to_boost(datetime_array[i].second);
		pvcop::types::formatter_datetime_us dtf_us(boost_datetime_format.c_str());

		//from string
		if (dtf_us.from_string(datetime_array[i].first.c_str(), out_array.data(), i)) {

			//to string
			static constexpr size_t str_len = 1024;
			char str[str_len];

			if(dtf_us.to_string(str, str_len, out_array.data(), i)) {
				if(strncmp(datetime_array[i].first.c_str(), str, datetime_array[i].first.size()!=0)){
					std::cerr << "Error: datetimes are not identical" << std::endl;
					std::cerr << "date: " << datetime_array[i].first << " - icu_format: " << datetime_array[i].second
										<< " - boost_format: " << boost_datetime_format << " - pvcop generate number: " << out_array.at(i)
										<< " - pvcop generate string: " << str << std::endl;
					return 1;
				}

			} else {
				std::cerr << "Fail to convert to datetime_us format (dtf_us.to_string)" << std::endl;
				std::cerr << " date: " << datetime_array[i].first << " - icu_format: " << datetime_array[i].second
					<< " - boost_format: " << boost_datetime_format << " - pvcop generate number: " << out_array.at(i)
					<< " - pvcop generate string: " << str << std::endl;
				return 1;
			}

		} else {
			std::cerr << "Fail to convert to datetime_us format (dtf_us.from_string)" << std::endl;
			std::cerr << " date: " << datetime_array[i].first << " - icu_format: " << datetime_array[i].second
				<< " - boost_format: " << boost_datetime_format << " - pvcop parser number: " << out_array.at(i) << std::endl;
			return 1;
		}
	}

	return 0;
}
