#ifndef PVKERNEL_CORE_PVLICENSE_H
#define PVKERNEL_CORE_PVLICENSE_H

#include <pvkernel/core/PVLogger.h>

#include <esisw-flex.h>
#include <string>
#include <stdexcept>

#include <sys/sysinfo.h>

namespace PVLicense {

    /**
     * Raise this exception when there is an error with licensing manager.
     */
    class NoFlexException : public std::exception
    {
        public:
            const char* what() const throw() { return "No License manager found"; }
    };

    /**
     * Raise this exception when The feature is missing
     */
    class NotAvailableFeatureException : public std::exception
    {
        public:
            NotAvailableFeatureException(std::string const& package, std::string const& feature): _msg(package + " " + feature + " is not available.")
        {
        }

            const char* what() const throw() { return _msg.c_str(); }

        public:
            std::string _msg;
    };

    /**
     * Init licensing system.
     */
    class RAII_InitLicense
    {
        public:
            RAII_InitLicense()
            {
                flexSetDisplayFunctionType(DISPLAY_MESSAGE_UNIX);
                if (flexInitialise() < 0)
                {
                    throw NoFlexException();           
                }
            }

            ~RAII_InitLicense()
            {
                flexRelease();
            }
    };

    /**
     * Hold a feature to handle token during use.
     */
    class RAII_LicenseFeature {
        public:
            RAII_LicenseFeature(std::string const& package, std::string const& feature) : _package(package), _feature(feature)
        {
            if (flexCheckFeature(_package.c_str(), _feature.c_str(), 2016, 1, LM_CO_NOWAIT, LM_DUP_NONE, DISPLAY_MESSAGE_UNIX)) {
                throw NotAvailableFeatureException(_package, _feature);
            }
        }

            static bool is_available(std::string const& package, std::string const& feature)
            {
                return flexAllowedFeature(package.c_str(), feature.c_str(), 2016, 1, LM_DUP_NONE);
            }

            ~RAII_LicenseFeature()
            {
                if(flexReleaseFeature(_package.c_str(), _feature.c_str())) {
                    std::string err = std::string("Token for package : ") + _package + " and feature : " +
                            _feature + " was not released.";
                    PVLOG_ERROR(err.c_str());
                }
            }

        private:
            std::string _package;
            std::string _feature;

    };

    /**
     * Check if the quantity of Ram provided by the license is respected.
     */
    void check_ram() {
        struct sysinfo info;
        sysinfo(&info);

        size_t status = flexGetLimitedValue ("II", "INSPECTOR", "MAXMEM");
        if (info.totalram > (status * 1024 * 1024 * 1024)) {
            throw NotAvailableFeatureException("INENDI Inspector", std::to_string(status) + " Go of ram");
        }
    }

    /**
     * Get number of days remaining in the license file.
     */
    int get_remaining_days() {
        return flexExpireDays("II","INSPECTOR");
    }

}

#endif
