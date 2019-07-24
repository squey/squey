cd dist

install -v -m755 Linux*/lib/*.so              /app/lib
install -v -m644 Linux*/lib/{*.chk,libcrmf.a} /app/lib

install -v -m755 -d                           /app/include/nss
cp -v -RL {public,private}/nss/*              /app/include/nss

install -v -m755 Linux*/bin/{certutil,nss-config,pk12util} /app/bin
install -v -m644 Linux*/lib/pkgconfig/nss.pc  /app/lib/pkgconfig
