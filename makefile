DNSMain.out: DNSMain.o problemdata.o derivative.o param.o dnscudawrapper.o
	nvcc -o DNSMain.out obj/DNSMain.o obj/problemdata.o obj/derivative.o obj/param.o obj/dnscudawrapper.o

DNSMain.o: src/DNSMain.cpp src/problemdata.h src/param.h
	g++ -c src/DNSMain.cpp -o obj/DNSMain.o -DUSE_CUDA
	
problemdata.o: src/problemdata.cpp src/problemdata.h src/derivative.h src/param.h
	g++ -c src/problemdata.cpp -o obj/problemdata.o 
	
derivative.o: src/derivative.cpp src/derivative.h src/param.h
	g++ -c src/derivative.cpp -o obj/derivative.o
	
param.o: src/param.cpp src/param.h
	g++ -c src/param.cpp -o obj/param.o 
	
dnscudawrapper.o:	src/dnscudawrapper.cu src/dnscudawrapper.h src/problemdata.h
	nvcc -c src/dnscudawrapper.cu -o obj/dnscudawrapper.o -arch sm_20
	
clean:
	rm *.o xel2d.out


