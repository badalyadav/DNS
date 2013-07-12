/*
 * Stores the parameters 
 */

#ifndef __PARAM_H__
#define __PARAM_H__

#include <stdio.h>
#include <fstream>
using namespace std;

	#define TAG "DefaultKernel_blockSize6" //tag that appends at the end of csv file names

	typedef double ptype;	//precision type
	
	#define INIT_FILE_NAME "ProblemDataFiles/Init_hom_iso_binary63.dat"
	#define PI 3.14159265359
		
		//the program stops if it either hits the target time or target iteration count
	#define TARGET_TIME 1
	#define TARGET_ITER 10
	
	//constants related to problem
	#define GAMMA 	1.4
	#define R 		1.0		//gas constant
	#define Cp 		3.5		//specific heat at constant pressure
	#define Pr 		0.72	//prantal number
	
	//short-hand macros
	#define FOR(i, n) for(int i=0; i<n; i++)
	#define I ((z)*N*N + (y)*N + x)
	#define WRAP(x) ((x+10*N)%N)
	#define Iw(i, j, k) ((WRAP(z+k))*N*N + (WRAP(y+j))*N + WRAP(x+i))
	
	//macros for declaring and deleting dynamic memory space for problem variables
	#define MAKE(var) var = new ptype[Np]
	#define KILL(var) delete [] var
	#define MAKE5(var) MAKE(var[0]); MAKE(var[1]); MAKE(var[2]); MAKE(var[3]); MAKE(var[4]);
	#define KILL5(var) KILL(var[0]); KILL(var[1]); KILL(var[2]); KILL(var[3]); KILL(var[4]); 
	
	extern int N;	//Number of points in each directions
	extern int Np;	//total number of points
	
	
	//General functions to be used
	template <typename T> T max(T *arr)
	{
		//calculating max of an array
		if (Np > 0)
		{
			T max = arr[0];
			FOR(i, Np)
			{
				if(arr[i]>max)
					max = arr[i];
			}
			return max;		
		}
		else
			return -1;
	}
	

	template <typename T> T sum(T *arr)
	{
		//calculating sum of an array
		T sum = 0;
		FOR(i, Np)	sum += arr[i];
		return sum;	
	}
	
	template <typename T> void print3DArray(T *arr)	//print a 3D array
	{
		//calculating sum of an array
		FOR(z, N)
		{
			FOR(y, N)
			{
				printf("(Z=%d, Y=%d)\n", z, y);
				FOR(x, N)
				{
					printf("%f\t", (float)arr[I]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}
	

class TimeRecord
{
public:
	int N;
	float totalCPUTime;
	float totalGPUTime;
	float memCopy1Time, memCopy2Time;	//host to device and device to host
	float kernel1Time;
	
	TimeRecord()
	{
		this->N = 0;
	}
	
	~TimeRecord()
	{
		string fileName = "Results/Time_";
		fileName = fileName + TAG + ".csv";
		fstream timeFile(fileName.c_str(), ios::out | ios::app);
		if(timeFile.tellg() == 0)
			timeFile<<"N, Time Steps, Total CPU Time, Total GPU Time, H2DCopyTime, D2HCopyTime, Kernel Time, Actual Speedup, Effective Speedup\n";
		timeFile<<N<<", "<<TARGET_ITER<<", "<<totalCPUTime<<", "<<totalGPUTime<<", "<<memCopy1Time<<", "<<memCopy2Time<<", "<<kernel1Time<<", "<<totalCPUTime/kernel1Time<<", "<<totalCPUTime/totalGPUTime<<"\n";
		timeFile.close();
	}
};

extern TimeRecord timeRecord;
	
#endif
