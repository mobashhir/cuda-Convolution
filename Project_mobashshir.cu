#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <vector>
#include <time.h>
#include <algorithm>	
#include <sstream>

using namespace std;

const double PI = 3.141592653589793238463;                                        

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define BLOCK_WIDTH 32 																		//predifined block width for uniform use across the code
#define BLOCK_HEIGHT 32																		//predifined block height for uniform use across the code
																							// 32 x 32 as the max threads suppoted are 1024
struct RGB
{		
   char r, g, b;																			//char values to store the three channel RGB PPM image
};

struct ImageRGB
{
	int w, h;																				//width and height of the Image
	vector<RGB> data;   																	//vector of RGB structure as it is easier to resize the vector for
};																							//different image sizes

struct LoGKernal
{												
	vector<float> kernal; 																	//Laplacian of Gaussian Kernal structure that stores one karnel
};																							//with the float values


struct LoGKernalsList
{
	vector<LoGKernal> KernalIndex;															//Laplacian of Gaussian Kernal list structure stores multiple kernals
};																							//of different sizes as they are required for scale space computation

void eat_comment(ifstream &f)   															// to remove comment from line and jump to EOF
{
	char linebuf[1024];																		
	char ppp;
	while (ppp = f.peek(), ppp == '\n' || ppp == '\r')										//parsing through the lines
		f.get();
	if (ppp == '#')																			//checking if the character is a #
		f.getline(linebuf, 1023);															//jumping to the end of the line
}

void load_ppm(ImageRGB &img, string &name) 													//function to load the PPM image
{
	ifstream file(name.c_str(), ios::binary);												//using streams to read the file in binary format
	if (file.fail())																		//throw error on screen if file opening fails
	{
		cout << "Could not open file: " << name << endl;
		return;
	}

	string imageFormat;																		//stores the Image format
	int bits = 0;

	file >> imageFormat; 																	//Reads the image format from the PPM input image file

	eat_comment(file);																		//removes comments if any in the input file
	file >> img.w; 																			// get width
	file >> img.h; 																			// get height
	file >> bits; 																			// get the binary values

	if (imageFormat != "P6" || img.w < 1 || img.h < 1 || bits != 255)   					//error check, only reads P6 types of images and throws error
	{
		cout << "Please select correct Image" << endl;
		file.close();
		return;
	}

	img.data.resize(img.w * img.h);															//resizing image to the size of the imput image

	file.get();																				
	file.read((char*)&img.data[0], img.data.size() * 3);									//reading the file into the image structure

	file.close();																			//always close the file
}

void store_ppm(ImageRGB &img, string &name)
{
	ofstream file(name.c_str(), ios::binary);												//create stream to write the file  
	if (file.fail())																		//throw error if creating the file stream fails	
	{
		cout << "Could not store file: " << name << endl;
		return;
	}

	string imageFormat = "P6";																//stores only P6 format..first line of the PPM is format

	file << imageFormat << "\n"; 															//store the image format
	file << img.w << " "; 																	// set width
	file << img.h << endl; 																	// set height
	file << "255" << endl; 																	// Set bits.. always 225

	file.write((char*)&img.data[0], img.data.size() * 3);									//writing the file from the image passed to the function

	file.close();																			//always close the file
}

static void HandleError(cudaError_t err, const char *file, int line) {  					//error handling for the cuda errors
	if (err != cudaSuccess) {																//anything other than successs status is printed on the screen
		std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << endl;
	}
}

vector<float> createLoGKernal(int sigma, int sizeOfKernal)									//CPU function that creates the Laplacian of Gaussian kernal
{
	vector<float> lKernal;																	//creating object of the kernal struct
	lKernal.resize(sizeOfKernal);															//resing to the passed size of the kernal

	float sum = 0.0f;
	float c1 = (-1 / (PI * pow(sigma,4)) );													//calcuating the value of the first component of the equation
	 																						//refer the report to see the entire equation
	int index = 0;
	for (int i = -sizeOfKernal / 2; i < sizeOfKernal / 2 + 1; i++)							//loop to calculatet the values of the kernal
	{
		float c2 = ( 1 - ( (i*i) / (2 * sigma * sigma)));									//calcuating the value of the second component of the equation
		float c3 = exp( (-1 *i*i) /(2 * sigma * sigma) );									//calcuating the value of the third component of the equation
		lKernal[index] = c1 * c2 * c3;														//evaluate the Laplacian of Gaussian kernal
		
		sum+=lKernal[index]; 																//calculating the sum to normalize the kernal later
		index++;																			//index of the kernal array	
	}

     
    for (int i = 0; i < lKernal.size(); i++)												//Normalizing the kernal distribution
        lKernal[i] /= sum;

	return lKernal;   																		
}
 
void medianFilterCPU(char *Input_Image, char *Output_Image, int Image_Width, int Image_Height) //CPU implementation of the median filter 
{
	char RNeighbor[9];																		//R values of the neighbouring pixels
    char GNeighbor[9];																		//G values of the neighbouring pixels
    char BNeighbor[9];																		//B values of the neighbouring pixels	

    int iterator;

    for(int x = 1; x < Image_Width - 1; x++)												//looping through the width of the image.. -1 for the edge case
    	for(int y = 1; y < Image_Height - 1; y++)											//looping through the height of the image.. -1 for the edge case
    	{	
    	iterator = 0;
    	for (int r = x - 1; r <= x + 1; r++) {												//finding the next neighbour pixels along the x axis	
        	for (int c = y - 1; c <= y + 1; c++) {											//finding the next neighbour pixels along the y axis	

        		size_t innerIndex = (c * Image_Width + r) * 3;								//getting the 1 D index of the pixel location

            	RNeighbor[iterator] = Input_Image[innerIndex + 0];							//storing the neighboring red pixels
            	GNeighbor[iterator] = Input_Image[innerIndex + 1];							//storing the neighboring green pixels
            	BNeighbor[iterator] = Input_Image[innerIndex + 2];							//storing the neighboring blue pixels

            	iterator++;
        	}
    	}

   		 for (int i=0; i<5; ++i) 															//using bubble sort to sort the 9 pixels
    	{
       	int minvalR=i;																		//store red value in the temporary pixel..easier access..clearer code
        int minvalG=i;																		//store green value in the temporary pixel..easier access..clearer code
        int minvalB=i;																		//store blue value in the temporary pixel..easier access..clearer code

        for (int l=i+1; l<9; ++l) 
        {
        	if (RNeighbor[l] < RNeighbor[minvalR]) 											//condition to check the red minimum from the array
        		minvalR=l;

        	if (GNeighbor[l] < GNeighbor[minvalG])											//condition to check the green minimum from the array 
        		minvalG=l;
        	
        	if (BNeighbor[l] < BNeighbor[minvalB]) 											//condition to check the blue minimum from the array		
        		minvalB=l;
        }
        																					// --- Put found minimum element in its place
        char tempR = RNeighbor[i];															//swap using the temporary variable
        RNeighbor[i]=RNeighbor[minvalR];													//if the min red value is found
        RNeighbor[minvalR]=tempR;

        char tempG = GNeighbor[i];															//swap using the temporary variable
        GNeighbor[i]=GNeighbor[minvalG];
        GNeighbor[minvalG]=tempG;															//if the min green value is found
        
        char tempB = BNeighbor[i];															//swap using the temporary variable
        BNeighbor[i]=BNeighbor[minvalB];
        BNeighbor[minvalB]=tempB;															//if the min blue value is found
   	    }
    
    int out_index = (y*Image_Width+x )* 3; 													//calculating the result 1 D index

    Output_Image[out_index + 0]=RNeighbor[4]; 												//storing the middle value..the median value of the red channel	
    Output_Image[out_index + 1]=GNeighbor[4]; 												//storing the middle value..the median value of the green channel	
    Output_Image[out_index + 2]=BNeighbor[4]; 												//storing the middle value..the median value of the blue channel
    }

}


__global__ void LoGConvolutionFunction(char* gpu_result_img_ptr, char* gpu_orig_img_ptr, size_t w, size_t h, float* KernalValues, size_t K){
	
	extern __shared__ unsigned char S[];													//defined a shared memory pointer
																							//calculate the index into the output image
	size_t bxi = blockIdx.x * blockDim.x;													//upper left corner of the block
	size_t xi = bxi + threadIdx.x;															//x-index for the current thread
	size_t yi = blockIdx.y * blockDim.y + threadIdx.y;										//y-index for the current thread

	if(xi >= w   || yi >= h  ) 																//exit if the output pixel is outside of the output image
	return;

	if(xi >= w - K + 1 || yi >= h - K + 1) 													//for indexes greater than the kernal width..the edge cases
	{
		size_t inner = (yi * (w ) + xi) * 3;												//calculating the 1 D index the image

		gpu_result_img_ptr[inner + 0] = 0;													//change value to 0	for the edge pixel values
 		gpu_result_img_ptr[inner + 1] = 0;
		gpu_result_img_ptr[inner + 2] = 0;
		return;
	}

	size_t i = (yi * (w  ) + xi) * 3;														//calculating the 1 D index for the output image
																							//create registers for shared memory access
	size_t Sw = blockDim.x + K - 1;															//width of shared memory
	size_t syi = threadIdx.y ;																//shared memory y-index

	for(size_t sxi = threadIdx.x; sxi < Sw; sxi += blockDim.x){								//copy block and curtain data from global memory to shared memory
		S[(syi * Sw + sxi) * 3 + 0] = gpu_orig_img_ptr[(yi * w + bxi + sxi) * 3 + 0];       //copying to the shared memory
		S[(syi * Sw + sxi) * 3 + 1] = gpu_orig_img_ptr[(yi * w + bxi + sxi) * 3 + 1];
		S[(syi * Sw + sxi) * 3 + 2] = gpu_orig_img_ptr[(yi * w + bxi + sxi) * 3 + 2];
	}

	__syncthreads();																		//synchronize threads after the global->shared copy
	
	float3 sum;																				//allocate a register to store the pixel sum
	sum.x = sum.y = sum.z = 0.0f;															//intialize the register values to zero
	
	size_t ypart = (syi * Sw + threadIdx.x) * 3;
	 
	for(size_t kxi = 0; kxi < K; kxi++){ 													//for each element in the kernel
		
		if( sum.x > 255.0f)																	//if the values overflow
			sum.x = sum.x - 255;															//basically taking modulus
		else
			sum.x += ( S[ypart + kxi * 3 + 0] * KernalValues[kxi] );						//summming and adding..convoluting the image with the kernal

		if( sum.y > 255.0f)																	//if the values overflow
			sum.y = sum.y - 255;															//basically taking modulus
		else
			sum.y += ( S[ypart + kxi * 3 + 1] * KernalValues[kxi] );

		if( sum.z > 255.0f)																	//if the values overflow
			sum.z = sum.z - 255;															//basically taking modulus
		else	
			sum.z += ( S[ypart + kxi * 3 + 2] * KernalValues[kxi] );
	}
	 																						//output the result for each channel
	gpu_result_img_ptr[i + 0] = sum.x ;														//for the red channel
 	gpu_result_img_ptr[i + 1] = sum.y ;														//for the green channel
	gpu_result_img_ptr[i + 2] = sum.z ;														//for the blue channel
	
}

__global__ void medianFilterKernel(char *Input_Image, char *Output_Image, int Image_Width, int Image_Height) {

    char RNeighbor[9];																		//R values of the neighbouring pixels
    char GNeighbor[9];																		//G values of the neighbouring pixels
    char BNeighbor[9];																		//B values of the neighbouring pixels

    int iterator;

    const int x = blockDim.x * blockIdx.x + threadIdx.x;									//x dimension index of the threads
    const int y = blockDim.y * blockIdx.y + threadIdx.y;									//y dimension index of the threads
    
    if( (x >= (Image_Width - 1)) || (y >= Image_Height - 1) )								//if the thread indexes are greater than the image
    	 return;

    iterator = 0;
   for (int r = x - 1; r <= x + 1; r++) {											    	//finding the next neighbour pixels along the x axis	
        	for (int c = y - 1; c <= y + 1; c++) {											//finding the next neighbour pixels along the y axis	

        		size_t innerIndex = (c * Image_Width + r) * 3;								//getting the 1 D index of the pixel location

            	RNeighbor[iterator] = Input_Image[innerIndex + 0];							//storing the neighboring red pixels
            	GNeighbor[iterator] = Input_Image[innerIndex + 1];							//storing the neighboring green pixels
            	BNeighbor[iterator] = Input_Image[innerIndex + 2];							//storing the neighboring blue pixels

            	iterator++;
        	}
    	}

    for (int i=0; i<5; ++i) {

        // --- Find the position of the minimum element
        int minvalR=i;
        int minvalG=i;
        int minvalB=i;

        for (int l=i+1; l<9; ++l) 
        {
        	if (RNeighbor[l] < RNeighbor[minvalR]) 											//condition to check the red minimum from the array
        		minvalR=l;

        	if (GNeighbor[l] < GNeighbor[minvalG])											//condition to check the green minimum from the array 
        		minvalG=l;
        	
        	if (BNeighbor[l] < BNeighbor[minvalB]) 											//condition to check the blue minimum from the array		
        		minvalB=l;
        }

        // --- Put found minimum element in its place
        char tempR = RNeighbor[i];															//swap using the temporary variable
        RNeighbor[i]=RNeighbor[minvalR];													//if the min red value is found
        RNeighbor[minvalR]=tempR;

        char tempG = GNeighbor[i];															//swap using the temporary variable
        GNeighbor[i]=GNeighbor[minvalG];
        GNeighbor[minvalG]=tempG;															//if the min green value is found
        
        char tempB = BNeighbor[i];															//swap using the temporary variable
        BNeighbor[i]=BNeighbor[minvalB];
        BNeighbor[minvalB]=tempB;															//if the min blue value is found
    }

    int out_index = (y*Image_Width+x )* 3; 													//calculating the result 1 D index

    Output_Image[out_index + 0]=RNeighbor[4]; 												//storing the middle value..the median value of the red channel	
    Output_Image[out_index + 1]=GNeighbor[4]; 												//storing the middle value..the median value of the green channel	
    Output_Image[out_index + 2]=BNeighbor[4]; 												//storing the middle value..the median value of the blue channel
}

__global__ void medianFilterKernelSharedMem(char *Input_Image, char *Output_Image, int Image_Width, int Image_Height) {

    __shared__ char RNeighbor[BLOCK_WIDTH*BLOCK_HEIGHT][9]; 								//shared memory to store the neighbor red values
    __shared__ char GNeighbor[BLOCK_WIDTH*BLOCK_HEIGHT][9];									//shared memory to store the neighbor greeen values
    __shared__ char BNeighbor[BLOCK_WIDTH*BLOCK_HEIGHT][9];									//shared memory to store the neighbor blue values


    int iterator;

    const int x     = blockDim.x * blockIdx.x + threadIdx.x;								//x dimension index of the input threads
    const int y     = blockDim.y * blockIdx.y + threadIdx.y;								//y dimension index of the input threads
    const int tid   = threadIdx.y * blockDim.x + threadIdx.x;   							//thread index 

    if( (x >= (Image_Width - 1)) || (y >= Image_Height - 1) ) return;						//if threads are out of the image size range 

    																						//Fill shared memory
    iterator = 0;
    for (int r = x - 1; r <= x + 1; r++) {													//finding the next neighbour pixels along the x axis
        for (int c = y - 1; c <= y + 1; c++) {												//finding the next neighbour pixels along the y axis
        		
        	size_t innerIndex = (c * Image_Width + r) * 3;									//calculating the 1 D index of the neighboring values
           
            RNeighbor[tid][iterator] = Input_Image[innerIndex + 0];							//storing the neighboring red pixels
            GNeighbor[tid][iterator] = Input_Image[innerIndex + 1];							//storing the neighboring green pixels
            BNeighbor[tid][iterator] = Input_Image[innerIndex + 2];							//storing the neighboring blue pixels


            iterator++;
        }
    }
    
    __syncthreads();																		//synchronize the threads to complete the copy to shared memory
    
    for (int i=0; i<5; ++i) {

       int minvalR=i;
       int minvalG=i;
       int minvalB=i;

        for (int l=i+1; l<9; ++l) 
        	{
        		if (RNeighbor[tid][l] < RNeighbor[tid][minvalR]) 							//condition to check the red minimum from the array
        			minvalR=l;

        		if (GNeighbor[tid][l] < GNeighbor[tid][minvalG]) 							//condition to check the green minimum from the array
        			minvalG=l;
        	
        		if (BNeighbor[tid][l] < BNeighbor[tid][minvalB]) 							//condition to check the blue minimum from the array
        			minvalB=l;
			}

        // --- Put found minimum element in its place

        char tempR = RNeighbor[tid][i];
        RNeighbor[tid][i]=RNeighbor[tid][minvalR];											//swap the values using a temporary variable
        RNeighbor[tid][minvalR]=tempR;

        char tempG = GNeighbor[tid][i];
        GNeighbor[tid][i]=GNeighbor[tid][minvalG];											//swap the values using a temporary variable
        GNeighbor[tid][minvalG]=tempG;
        
        char tempB = BNeighbor[tid][i];
        BNeighbor[tid][i]=BNeighbor[tid][minvalB];											//swap the values using a temporary variable
        BNeighbor[tid][minvalB]=tempB;


    }
     __syncthreads();
   
    int out_index = (y * Image_Width + x ) * 3; 											//calculating the result 1 D index

    Output_Image[out_index + 0]=RNeighbor[tid][4]; 											//store the result in the result array from the shared memory	
    Output_Image[out_index + 1]=GNeighbor[tid][4]; 
    Output_Image[out_index + 2]=BNeighbor[tid][4]; 

    __syncthreads();																		//synchronize the threads upon compeltion

}

__global__ void blobMaximaKernal(size_t* BlobIx, int  sigma, int orig_h, int orig_w, char* img1, char* img2, char* img3){

	size_t bxi = blockIdx.x * blockDim.x;													//upper left corner of the block
	size_t xi = bxi + threadIdx.x;															//x-index for the current thread
	size_t yi = blockIdx.y * blockDim.y + threadIdx.y;										//y-index for the current thread

	if(xi >= orig_w || yi >= orig_h ) 														//exit if the output pixel is outside of the output image
		return;

	size_t i = (yi * orig_w + xi) * 3;														//1D index for the output image

	int widthJump = 0;																		//width offset index of the image in scale space(image above and below) 
	if (i/orig_w % 2 == 0 )
		widthJump = (i/orig_w) * 3;
	else 
		widthJump = (i/orig_w + 1 ) * 3;
																							//calcualting the maximum of the 26 neighbours of the pixel

	if(    										img2[i + 0] > img2[i + 3] 	  	  && img2[i + 0] > img2[i - 3]  		// both the side edges R value
		&& img2[i + 0] > img2[widthJump - 1] && img2[i + 0] > img2[widthJump - 4] && img2[i + 0] > img2[widthJump - 7]
		&& img2[i + 0] > img2[widthJump + 1] && img2[i + 0] > img2[widthJump + 4] && img2[i + 0] > img2[widthJump + 7]
		//checking the image below in the scale
		&& img2[i + 0] > img1[i + 0] 		 && img2[i + 0] > img1[i + 3] 		  && img2[i + 0] > img1[i - 3] 
		&& img2[i + 0] > img1[widthJump - 1] && img2[i + 0] > img1[widthJump - 4] && img2[i + 0] > img1[widthJump - 7]
		&& img2[i + 0] > img1[widthJump + 1] && img2[i + 0] > img1[widthJump + 4] && img2[i + 0] > img1[widthJump + 7]
		//checking the image above in the scale
		&& img2[i + 0] > img3[i + 0] 		 && img2[i + 0] > img3[i + 3] 		  && img2[i + 0] > img3[i - 3] 
		&& img2[i + 0] > img3[widthJump - 1] && img2[i + 0] > img3[widthJump - 4] && img2[i + 0] > img3[widthJump - 7]
		&& img2[i + 0] > img3[widthJump + 1] && img2[i + 0] > img3[widthJump + 4] && img2[i + 0] > img3[widthJump + 7]

																							//for the G channel checking all the 26 neighbours
		  									 && img2[i + 1] > img2[i + 4 ] 		  && img2[i + 1] > img2[i - 4]  		// both the side edges
		&& img2[i + 1] > img2[widthJump - 2] && img2[i + 1] > img2[widthJump - 5] && img2[i + 1] > img2[widthJump - 8]
		&& img2[i + 1] > img2[widthJump + 2] && img2[i + 1] > img2[widthJump + 5] && img2[i + 1] > img2[widthJump + 8]
		
		&& img2[i + 1] > img1[i + 1] 		 && img2[i + 1] > img1[i + 4] 		  && img2[i + 1] > img1[i - 4] 
		&& img2[i + 1] > img1[widthJump - 2] && img2[i + 1] > img1[widthJump - 5] && img2[i + 1] > img1[widthJump - 8]
		&& img2[i + 1] > img1[widthJump + 2] && img2[i + 1] > img1[widthJump + 5] && img2[i + 1] > img1[widthJump + 8]

		&& img2[i + 1] > img3[i + 1] 		 && img2[i + 1] > img3[i + 4] 		  && img2[i + 1] > img3[i - 4] 
		&& img2[i + 1] > img3[widthJump - 2] && img2[i + 1] > img3[widthJump - 5] && img2[i + 1] > img3[widthJump - 8]
		&& img2[i + 1] > img3[widthJump + 2] && img2[i + 1] > img3[widthJump + 5] && img2[i + 1] > img3[widthJump + 8]

																							//for the Blue channel checking all the 26 neighbours
		   									 && img2[i + 2] > img2[i + 5 ] 		  && img2[i + 2] > img2[i - 5]  // both the side edges
		&& img2[i + 2] > img2[widthJump - 3] && img2[i + 2] > img2[widthJump - 6] && img2[i + 2] > img2[widthJump - 9]
		&& img2[i + 2] > img2[widthJump + 3] && img2[i + 2] > img2[widthJump + 6] && img2[i + 2] > img2[widthJump + 9]
		
		&& img2[i + 2] > img1[i + 2] 		 && img2[i + 2] > img1[i + 5] 		  && img2[i + 2] > img1[i - 5] 
		&& img2[i + 2] > img1[widthJump - 3] && img2[i + 2] > img1[widthJump - 6] && img2[i + 2] > img1[widthJump - 9]
		&& img2[i + 2] > img1[widthJump + 3] && img2[i + 2] > img1[widthJump + 6] && img2[i + 2] > img1[widthJump + 9]

		&& img2[i + 2] > img3[i + 2] 		 && img2[i + 2] > img3[i + 5] 		  && img2[i + 2] > img3[i - 5] 
		&& img2[i + 2] > img3[widthJump - 3] && img2[i + 2] > img3[widthJump - 6] && img2[i + 2] > img3[widthJump - 9]
		&& img2[i + 2] > img3[widthJump + 3] && img2[i + 2] > img3[widthJump + 6] && img2[i + 2] > img3[widthJump + 9]
	)
	{
		BlobIx[i] = i;																		//if the pixel is the maxima, then store it
	}

	return;
}

int circle(int x, int y, int cx, int cy, int radius){										//function to calculate the circumference indexes of the circle

    if ((int)sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)) == radius)									//equation of the circle
        return 1;																			//returns true if the index lies on the circumference
    else
        return 0;																			//returns false if the index lies on the circumference
}

int isNearCircle(int x, int y, int cx, int cy, int radius){									//function to find if the index lies near to a given circle

    if ((int)sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)) <= (int)(radius*1.5) )						//equation of the circle with an increased raduis to reduce overlapping
        return 1;																			//returns true if the index lies near the circle
    else
        return 0;																			//returns false if the index is not near the circle
}

int main(int argc, char** argv) {
	
	if (argc != 6) 																			//There should be six command line arguments
	{
		cout << "Enter corrrect arguments as ./a.out image.ppm A B C D"<< endl;
		cout << " Float A = t used as sigma = exp(t) " << endl;
		cout << " Float B = increasing_factor used as t+=B" << endl; 
		cout << " Int   C = numberOfIterations" << endl;
		cout << " String D = blob or median" << endl;
		cout << " Ex1 : ./a.out applegreen.ppm 0 0.5 5 median" << endl << "Ex1 : ./a.out butterfly.ppm 0 0.5 5 blob" << endl ;
		return 1; //exit and return an error
	}
	 
	string InputImagefilename = argv[1];													//getting the input filename from the command line arguments
	float t = atof(argv[2]);																//getting the first sigma value in the scale space
	float increasingFactor = atof(argv[3]);													//the value with which the sigma is to be incremented in scale space
	int numberOfIterations = atoi(argv[4]);													//the number of the scales of the scale space
	string typeOfFilter = argv[5];															//type of filter to be applied to the image - Blob Filter/Median filter

	

	//Caluclating the sigma values for the number of times the LoG has to be applied
	int *sigmaList;                															//Declare pointer to type of array
	sigmaList = new int[numberOfIterations];												//dynamic allocation of the number of scales
	
	for(int i = 0; i < numberOfIterations; i++){											//calculating sigma in exponential factors
		sigmaList[i] = exp(t);
		t += increasingFactor;  
	}

	LoGKernalsList kList;																	//declaring list of kernals to store the different kernal for each sigma
	kList.KernalIndex.resize(numberOfIterations);											//resizing the kernal vector to the scales
	
	for(int i = 0; i < numberOfIterations; i++)												//creating Laplacian of Gaussian (LoG) Kernals		
	{
		int sizeOfKernal = 6 * sigmaList[i]; 												//calculate size of the kernal, keeping 99% of the variance
		if (sizeOfKernal % 2 == 0)															//kernal size should be odd
			sizeOfKernal++;
		kList.KernalIndex[i].kernal.resize(sizeOfKernal);
	
		kList.KernalIndex[i].kernal = createLoGKernal(sigmaList[i], sizeOfKernal);  		//call to create the laplacian of gaussian karnel
		
	}
	
	ImageRGB InputImg;  																	//original image
 
	load_ppm(InputImg, InputImagefilename);													//Loading the PPM input image
	 
	int count;																				//stores the number of CUDA compatible devices
	 HANDLE_ERROR(cudaGetDeviceCount(&count));												//get the number of devices with compute capability < 1.0
	if (count < 1)																			//throw error if no CUDA device is found	
	{
		cout << "No cuda device " << endl;
		return 1; 																			//exit and return an error
	}

	char* gpu_orig_img_ptr;
	
	HANDLE_ERROR(cudaMalloc(&gpu_orig_img_ptr, InputImg.h * InputImg.w * 3 * sizeof(char))); //memory allocation on the device for the original image 

																							//copying image to device
	HANDLE_ERROR(cudaMemcpy(gpu_orig_img_ptr, &InputImg.data[0] , InputImg.h * InputImg.w * 3 * sizeof(char), cudaMemcpyHostToDevice));

	////-----------------------------------------------------------------------/////
	////------------------------BEGENNING OF MEDIAN FILTERING------------------/////

	if(typeOfFilter == "median")
	{

	cout << "Executing in the CPU " << endl;
																							//CPU Implementation
	ImageRGB CPUmedianFImage;																//result image generated from the CPU execution
	CPUmedianFImage.h = InputImg.h ;	
	CPUmedianFImage.w = InputImg.w ;
	CPUmedianFImage.data.resize(CPUmedianFImage.h * CPUmedianFImage.w);						//resing the image vector

	const clock_t begin_time = clock();														//begin timer to calcualte the CPU execution time
																							//calling the function to execute on the CPU
	medianFilterCPU(&InputImg.data[0].r , &CPUmedianFImage.data[0].r , CPUmedianFImage.w, CPUmedianFImage.h);

	float cpuTime = ( clock () - begin_time ) * 1000 /  CLOCKS_PER_SEC ; 					//calcuating the execution time in milliseconds
	std::cout << "CPU execution time = " << cpuTime << " milliseconds";

	string CPUmedianfile = "Median_Filter_Image_CPU.ppm";
    store_ppm(CPUmedianFImage , CPUmedianfile);  											//storing the CPU image
	
																							//GPU Implementation 
	ImageRGB medianFImage;																	//resultant image from median filtering
	medianFImage.h = InputImg.h ;	
	medianFImage.w = InputImg.w ;
	medianFImage.data.resize(medianFImage.h * medianFImage.w);								//resizing the resultant image

	char* gpu_medianF_ptr;																	//pointer to the gpu result image
																							//memory allocation fro the result image
	HANDLE_ERROR(cudaMalloc(&gpu_medianF_ptr, medianFImage.h * medianFImage.w * 3 * sizeof(char)));
 
	dim3 threadsMedian(32, 32);																//create a square block of 1024 threads
	dim3 blocksMedian(medianFImage.w /threadsMedian.x + 1, medianFImage.h / threadsMedian.y + 1);	//calculate # of blocks

	cudaEvent_t start1, stop1, start2, stop2 ;												//declare a start and stop event
	cudaEventCreate(&start1);																//create both start events
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);																//create both stop events
	cudaEventCreate(&stop2);

	cudaEventRecord(start1);																//insert the start event into the stream
																							//call the kernal function with the above configuration	
    medianFilterKernel<<<blocksMedian,threadsMedian>>>(gpu_orig_img_ptr, gpu_medianF_ptr, medianFImage.w, medianFImage.h);

    cudaEventRecord(stop1);																	//insert the stop event into the stream
	cudaEventSynchronize(stop1);															//wait for the stop event, if it isn’t done
	float milliseconds1 = 0;																//declare a variable to store runtime
	cudaEventElapsedTime(&milliseconds1, start1, stop1);									//get the elapsed time	
	cout << "\nTime taken for simple Implementation: " << milliseconds1 << " milliseconds "<< endl;

	cudaEventRecord(start2);																//insert the start event into the stream
																							//Call the kernal using the shared memory implementation
    medianFilterKernelSharedMem<<<blocksMedian,threadsMedian>>>(gpu_orig_img_ptr, gpu_medianF_ptr, medianFImage.w, medianFImage.h);

	cudaEventRecord(stop2);																	//insert the stop event into the stream
	cudaEventSynchronize(stop2);															//wait for the stop event, if it isn’t done
	float milliseconds2 = 0;																//declare a variable to store runtime
	cudaEventElapsedTime(&milliseconds2, start2, stop2);									//get the elapsed time	
	cout << "\nTime taken for shared memory Implementation: " << milliseconds2 << " milliseconds "<< endl;

																						    //copying image from device to host
	HANDLE_ERROR(cudaMemcpy(&medianFImage.data[0], gpu_medianF_ptr, medianFImage.h * medianFImage.w * 3 * sizeof(char), cudaMemcpyDeviceToHost)); 

    string medianFilterFileName = "Median_Filter_Image.ppm";
    store_ppm(medianFImage , medianFilterFileName);  										//storing the GPU image
	
	cudaFree(gpu_orig_img_ptr);        														//freeing threads
	cudaFree(gpu_medianF_ptr);																//very essential to free the threads	

	}

	////-----------------------------------------------------------------------/////
	////--------------------------END OF MEDIAN FILTERING----------------------/////


	////---------------------------------------------------------------------/////
	////------------------------BEGENNING OF BLOB DETECTION------------------/////

	else if(typeOfFilter == "blob")
	{

	char** LoG_result_img_ptr; 												              	//Declare pointer to type of array
	LoG_result_img_ptr = new char*[numberOfIterations];										//Array of pointers of the kernal
	
	for(int j = 0; j < numberOfIterations; j++)												//Looping for each kernal for different sigma values	
	{
		float* LoGKernal_ptr;																//declare pointer to each kernal evrytime in the loop

		int kernalSize = kList.KernalIndex[j].kernal.size(); 								//resize this pointer to the kernal size for each sizma value

		HANDLE_ERROR(cudaMalloc(&LoGKernal_ptr, kernalSize * sizeof(float)));				//alloate the memory on the device for the kernal
		
																							//copying kernal to device
		HANDLE_ERROR(cudaMemcpy(LoGKernal_ptr, &kList.KernalIndex[j].kernal[0] , kernalSize * sizeof(float), cudaMemcpyHostToDevice));

		ImageRGB LoGImage;																	//object of the convolved image
		LoGImage.h = InputImg.h ;				
		LoGImage.w = InputImg.w ;
		LoGImage.data.resize(LoGImage.h * LoGImage.w);										//keeping same size of the input image..makes life easy for 
																							//finding the maxima, else its hell matching the indexes in scale space
																							//allocating the memory for each convolved image 
		HANDLE_ERROR(cudaMalloc(&LoG_result_img_ptr[j], LoGImage.h * LoGImage.w * 3 * sizeof(char)));

																							//create a CUDA grid configuration
		dim3 threads(32, 32);																//create a square block of 1024 threads
		dim3 blocks(LoGImage.w /threads.x + 1, LoGImage.h / threads.y + 1);					//calculate # of blocks

																							//calculate the required size of shared memory
		size_t Sbytes = ((threads.x + kernalSize - 1) * threads.y * sizeof(char)) * 3;
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, 0);													//get properties to check if there is enough shared memory
		if(props.sharedMemPerBlock < Sbytes){
			cout<<"ERROR: insufficient shared memory"<<std::endl;
	 		exit(1);
		}
																							//Calling the kernal function to convolve the image
	 	LoGConvolutionFunction <<<blocks, threads, Sbytes >>>(LoG_result_img_ptr[j], gpu_orig_img_ptr, InputImg.w, InputImg.h, LoGKernal_ptr, kernalSize);

																							//copying image from device to host
		HANDLE_ERROR(cudaMemcpy(&LoGImage.data[0], LoG_result_img_ptr[j], LoGImage.h * LoGImage.w * 3 * sizeof(char), cudaMemcpyDeviceToHost)); 

		std::stringstream sstm;
		sstm << "Convolved_Image_" << j << ".ppm";
		string LoGFileName = sstm.str();

		store_ppm(LoGImage, LoGFileName); 													//storing gpu image

		cudaFree(LoG_result_img_ptr);														//freeing the threads
		cudaFree(LoGKernal_ptr);
	}


	//Getting the maximum value of the pixel from the genreated images through the Laplacian of Gaussian kernal
	size_t* blob;

	int numberOfBLobs = numberOfIterations - 2;												//number of the scales in which the blob can be calculated
																							//the first sigma and the last sigma images will be included
																							//as they donot have the required 26 neighbors in scale space		
	int sizeofBLobImages = InputImg.w * InputImg.h * 3 ; 									//resizing to all the indexes of the images size

	size_t *blobIdx;
	blobIdx = new size_t[sizeofBLobImages * numberOfBLobs];									//dynmically allocating the number of blob images array

	memset(blobIdx, 0, sizeofBLobImages * numberOfBLobs * sizeof(size_t) ) ;				//setting all the indexes to zero

	int indexOfBLobCenters = 0 ;

 	for(int scales = 1; scales < numberOfIterations-1; scales++)							//looping through the scale space
 	{
 	
 	HANDLE_ERROR(cudaMalloc(&blob, InputImg.w * InputImg.h * 3 * sizeof(size_t) )); 		// mem allication  to get the max size for all possible blobs
 	HANDLE_ERROR(cudaMemset(blob, 0, InputImg.w * InputImg.h * 3 * sizeof(size_t) ) );      //setting the values to zero
	
 	dim3 threadsBlob(32, 32);																//create a square block of 1024 threads
	dim3 blocksBlob(InputImg.w /threadsBlob.x + 1, InputImg.h / threadsBlob.y + 1);			//calculate # of blocks

																							//Call the function to calculate the maxima in scale space
	blobMaximaKernal <<<blocksBlob, threadsBlob >>>(blob, sigmaList[scales], InputImg.h, InputImg.w,
													 LoG_result_img_ptr[scales - 1] , LoG_result_img_ptr[scales], LoG_result_img_ptr[scales + 1]);
			
																							//copying image from device to host
	HANDLE_ERROR(cudaMemcpy(&blobIdx[indexOfBLobCenters], blob, InputImg.h * InputImg.w * 3 * sizeof(size_t), cudaMemcpyDeviceToHost)); 

	int raduis = 1.5 * sigmaList[scales];													//radius of the blob is given by this formula

	cout << "Removing Duplicates of the same size " << raduis << endl;

	int county = 0;																			//counter index
	for(int i=0; i< InputImg.w * InputImg.h * 3; i++)
	{
		if(blobIdx[indexOfBLobCenters + i] != 0)											//for each blob center
		{	
			int cy = (blobIdx[indexOfBLobCenters + i] / 3) % InputImg.w;					//calculate the y index in 2D space of the blob 
        	int cx = (blobIdx[indexOfBLobCenters + i] / 3) / InputImg.w;					//calculate the x index in 2D space of the blob

        	for(int k = i+raduis ; k < InputImg.w * InputImg.h * 3 ; k++)
        	{					//for all pixels in the image after this index
				int kcy = (blobIdx[indexOfBLobCenters + k] / 3) % InputImg.w;				//calculate the y index in 2D space of the image
        		int kcx = (blobIdx[indexOfBLobCenters + k] / 3) / InputImg.w;				//calculate the x index in 2D space of the image
				if(isNearCircle(cx, cy, kcx, kcy, raduis))									//call the function to check if the index is near the blob
					{
						blobIdx[indexOfBLobCenters + i] = 0;								//remove that blob..prevent overlapping blobs
						//if(scales == (numberOfIterations-2))
						//for(int pk = 0; pk < numberOfIterations - 2; pk++){
						//blobIdx[indexOfBLobCenters + k - ( pk* sizeofBLobImages)] = 0;
					}
			} 
		}
		county++;	
	}

	indexOfBLobCenters += sizeofBLobImages;													//incrementing the index of the blob pixels by the image size
	}

	cout << "Removing concentric circles of different sizes" << endl;						
																							//Removing overlapping circles of different sizes..concentric circles
	for(int i=(sizeofBLobImages * numberOfBLobs) - sizeofBLobImages ; i< sizeofBLobImages * numberOfBLobs; i++)
	{
		int raduis = 1.5 * sigmaList[numberOfBLobs + 1];									//formula to get the radius of the blob

		if(blobIdx[i] != 0)																	//for all blob indexes not zero
		{	
			int cy = (blobIdx[i] / 3) % InputImg.w;											//calculate the y index in 2D space of the blob
        	int cx = (blobIdx[i] / 3) / InputImg.w;											//calculate the x index in 2D space of the blob

        	for(int k = i+1 ; k < InputImg.w * InputImg.h * 3 - i ; k++){
				int kcy = (blobIdx[k] / 3) % InputImg.w;									//calculate the y index in 2D space of the blob in other scale space
        		int kcx = (blobIdx[k] / 3) / InputImg.w;									//calculate the x index in 2D space of the blob in other scale space
			
			if(isNearCircle(cx, cy, kcx, kcy, raduis)){										//if the blobs are near over lapping each other
				 
				for(int pk = 0; pk < numberOfBLobs-1; pk++){								//set those indexes to xero for smaller scales 
						blobIdx[i - pk*sizeofBLobImages] = 0;
					}
				} 
			}		 
		}
	}

																							//drawing the circles on the blobs
	cout << "Drawing circles ... this will take some time if you have taken a very large image" << endl; 
    for(int index = 0 ; index < sizeofBLobImages * numberOfBLobs; index++)
    {
		int raduis = 1.5 * sigmaList[(index / sizeofBLobImages) + 1];						//formula to get the raduis of the circle

		if(blobIdx[index] != 0)																//for all indexes with blobs
		{	
			int cy = (blobIdx[index] / 3) % InputImg.w;										//calculate the y index in 2D space of the blob
        	int cx = (blobIdx[index] / 3) / InputImg.w;										//calculate the x index in 2D space of the blob
        	
    		for(int x=0; x<InputImg.h; x++)													//for each x index of the image
     			for(int y=0; y<InputImg.w; y++)												//for each y index of the image
        			if (circle(x, y, cx, cy, raduis))										//if the x and y cordinates are in the circumference of the blob
            		{
            		InputImg.data[x * InputImg.w + y].r = 255;								//keep only the red values
            		InputImg.data[x * InputImg.w + y].g = 0;								//to draw a red circle aroud the blob
            		InputImg.data[x * InputImg.w + y].b = 0;
        			}
    	}
    }

	string blobResultName = "Result_Blob_Image.ppm"; 	

	store_ppm(InputImg, blobResultName); 													//storing the blob image
	
	cudaFree(blob);																			//freeing the threads  	
	cudaFree(gpu_orig_img_ptr);
	cudaFree(LoG_result_img_ptr);
	}
		
	////----------------------------------------------------------------
	////------------------------END OF BLOB DETECTION------------------

    cudaFree(gpu_orig_img_ptr);																//freeing the original image

	return 0;
}

