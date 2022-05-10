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
Taking a polynomial smoothing of the trajectory and locating the inflection point we have
![](images/smoothed_data_derivatives.png)
Visualizing the inflection point on the smoothed trajectory
![](images/inflection_point_trajectory.png)
We can see that the inflection point is located approximately where the second derivative changes of sign, but this point is not the point where the cars begin to go backwards, where are going to use the point in the first derivative where it changes sign, this point corresponds to the local maximum and is where the trajectory of the cars begin to go backwards.

