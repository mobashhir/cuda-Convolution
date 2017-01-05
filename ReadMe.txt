Instructions on how to run the code

The code will either run the Median filter or the blob detector as specified in your command line

1-Compile
nvcc project_mobashshir.cu

2- Execute

For blob detection
./a.out image.ppm A B C D blob

A = t used as sigma = exp(t)
Float B = increasing_factor used as t+=B" << endl; 
Int   C = numberOfIterations" << endl;
String D = blob or median" << endl;
Example : ./a.out butterfly.ppm 0 0.5 5 blob



For median filter
./a.out ImageName.ppm 0 0.5 5 median

Example: ./a.out applegreen.ppm 0 0.5 5 median


