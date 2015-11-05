#!/usr/bin/python
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

import os
import re
import hashlib

from Crypto.Cipher import AES

#import base64
#import binhex
import binascii

# We use AES encryption using the customer name as a key
# this is simply used to identify a leaked binary.
# Since AES requires a 16, 24 or 32 bits key we then
# need to add padding.
CUSTOMER_KEY="ESI Group"
# Add automatic padding
lcustomer = len(CUSTOMER_KEY)
# We take the first 32 chars, that is enough
if (lcustomer >= 32):
    CUSTOMER_KEY=CUSTOMER_KEY[:32]
else:
    i = lcustomer
    while i < 32:
        CUSTOMER_KEY=CUSTOMER_KEY + "."
        i += 1

aes_obj = AES.new(CUSTOMER_KEY)

OBFUSCATE_HEADER="src/include/inendi/api-obfuscate.h"
OBFUSCATE_TMP=OBFUSCATE_HEADER + ".tmp"

os.system("grep LibExport src/include/inendi/*.h |grep -v define > %s" % (OBFUSCATE_TMP))
f_tmp = open(OBFUSCATE_TMP, "r")
f_header = open(OBFUSCATE_HEADER, "w")

extract_function_pattern = re.compile(".*(inendi_.*)\(.*")
struct_pattern = re.compile(".*_t .*")
for line in f_tmp:
    m  = extract_function_pattern.match(line)
    if m:
        function = m.group(1)
        struct = struct_pattern.match(function)
        if struct:
            # We don't want to match structures!
            continue

    else:
        continue

    function_hash = hashlib.sha256(function).hexdigest()
    encrypted_hash = aes_obj.encrypt(function_hash)
    hexval = binascii.hexlify(encrypted_hash)
    # print hexval
    f_header.write("#define " + function + " P" + hexval + "\n")
    f_header.write("#define " + function + "_string" + " \"P" + hexval + "\"\n")

f_tmp.close()
f_header.close()

os.unlink(OBFUSCATE_TMP)

