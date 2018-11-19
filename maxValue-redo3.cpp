//with image mask test column element groups of 4!!!  Top-1
//Version 2&3 (C and SSE with index) on 2k*1k*20 array.
//Extract index from probability
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <ctime>
#include <stdlib.h>
#include <bitset>
#include <climits>
//#include <xmmintrin.h>
//#include <emmintrin.h>
#include <smmintrin.h>
using namespace std;


const int numRows = 2;
const int numCols = 4;
const int classes = 20;


void argmax_C(float ***prob)
{
	for (int i = 0; i < numRows; i++)
    {
        for(int j = 0; j < numCols; j++)
        {
            float max = prob[i][j][0];

            for (int k = 0 ; k < classes; k++)
            {
                if (max < prob[i][j][k])
                {
                    max = prob[i][j][k];
                }
            }

            //cout << "largest value in r,c: " << i << "," << j << " = " << max << endl;

        }
    }

}

void argmax_sse(float ***prob)
{	
	for (int i = 0; i < numRows; i++)
    {
        for(int j = 0; j < numCols; j+=4)
        {     	
        	//__m128 aMaxVal = _mm_setr_ps(prob[i][j][0], prob[i][j+1][0], prob[i][j+2][0], prob[i][j+3][0]);
            //__m128 aMaxVal = _mm_loadu_ps(&prob[i][j][0]); // load the first 4
            __m128 aMaxVal = _mm_setzero_ps();
            /*
            cout << "j loop: " << endl;
            for (int q = 0; q < 4; q++)
            {
                cout <<  aMaxVal[q] << " ";
            }
            cout << endl;
            */
            for(int k = 0; k < classes; k++)
        	{
                __m128 cur = _mm_setr_ps(prob[i][j][k], prob[i][j+1][k], prob[i][j+2][k], prob[i][j+3][k]);
                //__m128 cur = _mm_loadu_ps(&prob[i][j][k]);
                /*
                cout << "k loop: " << endl;
                for (int q = 0; q < 4; q++)
                {
                    cout << cur[q] << " ";
                }
                cout << endl;
                */
                aMaxVal = _mm_max_ps(aMaxVal, cur);
                
               
        	}
            /*
            for (int q = 0; q < 4; q++)
            {
                cout <<  "max SSE = " << aMaxVal[q] << endl;
            }
            */

            //TO EXTRACT INDEX FROM PROBABILITY
            unsigned char out[4]; 
            __m128i mask = _mm_set1_epi32(1023);
            __m128i vMaxIndex =  _mm_and_si128((__m128i)aMaxVal, mask);
            //float ins[4] = {10.4, 10.6, 120, 100000};
            //__m128 x = _mm_load_ps(ins);       // Load the floats
            //__m128i y = _mm_cvtps_epi32(vMaxIndex);    // Convert them to 32-bit ints
            vMaxIndex = _mm_packus_epi32(vMaxIndex, vMaxIndex);        // Pack down to 16 bits
            vMaxIndex = _mm_packus_epi16(vMaxIndex, vMaxIndex);        // Pack down to 8 bits
            *(int*)out = _mm_cvtsi128_si32(vMaxIndex); // Store the lower 32 bits
            //*out = _mm_extract_epi32(vMaxIndex,0);
            
            printf("%d\n", out[0]);
            printf("%d\n", out[1]);
            printf("%d\n", out[2]);
            printf("%d\n", out[3]);
            
		    
	   }
    }
}


int main()
{

    float ***prob = new float**[numRows];
	
	prob = new float **[numRows]();
    for (int i = 0; i < numRows; i++)
    {
        prob[i] = new float *[numCols]();
        for (int j = 0; j < numCols; j++)
            prob[i][j] = new float [classes]();
    }
    //generating random 1000 floats between 0.0 - 1.0
    srand(time(NULL));
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            for (int k = 0; k < classes; k++)
            {
                prob[i][j][k] = float(rand()) / (float(RAND_MAX) + 1.0);
            }
        }
    }

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            //cout << "r,c = " << i << "," << j << ":" << endl; 
            for (int k = 0; k < classes; k++)
            {
                union
                {
                    float input;   // assumes sizeof(float) == sizeof(int)
                    int   output;
                }   data;

                data.input = prob[i][j][k];

                bitset<sizeof(float) * CHAR_BIT>   maxBinary(data.output); 

                bitset<sizeof(float) * CHAR_BIT>   maxIndexBin(k); 
                bitset<32> mask = 0XFFFFFC00;
                maxBinary = maxBinary & mask;

                maxBinary = maxBinary | maxIndexBin;
                prob[i][j][k] = reinterpret_cast<float &>(maxBinary);
                //cout << "binary of max after masking= "<< maxBinary << std::endl;

            }
        }
    }

    //Displaying the sample array
    /*
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            cout << "r,c = " << i << "," << j << ":" << endl; 
            for (int k = 0; k < classes; k++)
            {
                cout << prob[i][j][k] << " ";
            }
            cout << endl;
        }
       cout << endl << endl; 
    }
    
    */
    //argmax using C++ (reference code)
    clock_t begin = clock();

    argmax_C(prob);
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;
    cout << "normal time = " << elapsed_secs << endl;


	//argmax with SSE4.1

	clock_t start1 = clock();
	
	argmax_sse(prob);
    
	clock_t end1 = clock();
  	double elapsed_secs1 = double(end1 - start1) / CLOCKS_PER_SEC *1000;
  	cout << "sse time with masking = " << elapsed_secs1 << endl;


  	for (int i = 0; i < numRows; i++)
	{
	    for (int j = 0; j < numCols; j++)
	        delete[] prob[i][j];
	    delete[] prob[i];
	}
	delete[] prob;
	return 0;

}


