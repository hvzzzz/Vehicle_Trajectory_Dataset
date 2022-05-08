# Vehicle_Trayectory_Dataset
## Experiment Log
Finally we got our traffic space time diagrams but they look like this :'v
![](images/raw_spacetime.png)
![](images/raw_spacetime_zoom.png)
Taking the second derivative we get the acceleration.
![](images/accelerations.png)
Taking the EMD to the acceleration signal 
![](images/acce_EMD.png)
In order to clean the noises in the trajectories we filtered the first, second and third intrinsic mode functions and added the filtered functions to the rest of the IMF, but the results were not as expected. 
![](images/filtered_EMD.png)
We also tried another approach that applies a Gaussian Kernel directly to the trajectories
![](images/g_kernel_pos.png)


