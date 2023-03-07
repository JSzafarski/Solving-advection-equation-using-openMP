#include <stdio.h>
#include <math.h>
#include <omp.h>
int main(){

  /* Grid properties */
  const int NX=1000;    // Number of x points reduce this to 100 now but change this to 1000 when submiting work
  const int NY=1000;    // Number of y points
  const float xmin=0.0; // Minimum x value
  const float xmax=30.0; // Maximum x value
  const float ymin=0.0; // Minimum y value
  const float ymax=30.0; // Maximum y value
  
  /* Parameters for the Gaussian initial conditions */
  const float x0=3.0;                    // Centre(x)
  const float y0=15.0;                    // Centre(y)
  const float sigmax=1.00;               // Width(x)
  const float sigmay=5.00;               // Width(y)
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boudnary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper bounary
  
  /* Time stepping parameters */
  const float CFL=0.55;   // CFL number ,lowered number increases accuracy and stability but it also required more time to compute as it's more accurate
  const int nsteps=800; // Number of time steps

  /* Velocity */
        float velx=1.00; // Velocity in x direction
  const float vely=0.00; // Velocity in y direction
  
  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u

  float x2;   // x squared (used to calculate iniital conditions)
  float y2;   // y squared (used to calculate iniital conditions)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);

  //values used to calculate the vertical shear
  float zZero = 1.0; //paramaters for the shear logarithmic equation
  float uStar= 0.2;
  float kValue = 0.41;
  float velocity_constant = uStar/kValue; //showing this explicity
  
  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(velx) / dx) + (fabs(vely) / dy));
  
  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

    //the two loops can be parallelised without a problem
    #pragma omp parallel default(none) shared(x,y,dx,dy,NX,NY) //always use deafult(none) to force the programmer to explicity decalre shared and provate variables
    {
        /*** Place x points in the middle of the cell ***/
        /* LOOP 1 */
        for (int i=0; i<NX+2; i++){
            x[i] = ( (float) i - 0.5) * dx;
        }

        /*** Place y points in the middle of the cell ***/
        /* LOOP 2 */
        for (int j=0; j<NY+2; j++){
            y[j] = ( (float) j - 0.5) * dy;
        }
    }
  
    /*** Set up Gaussian initial conditions ***/
    /* LOOP 3 */
    //here we can use collapse keyword as we have a inner loop thats not related to the outerloop by iteration counts
    #pragma omp parallel for default(none) shared(u,NX,NY,x,y,x0,y0,sigmax2,sigmay2) private(x2,y2) //we have to make x and y2 private to not cause unexpected results due to race conditions
    for (int i=0; i<NX+2; i++){
        for (int j=0; j<NY+2; j++){
          x2      = (x[i]-x0) * (x[i]-x0);
          y2      = (y[j]-y0) * (y[j]-y0);
          u[i][j] = exp( -1.0 * ((x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2))));
        }
    }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4 */
  for (int i=0; i<NX+2; i++){ //cannot parallelise because this is data output so it has to be delivered in a specific order an multithreading doesnt gurantee that
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);
  
  /*** Update solution by looping over time steps ***/
  /* LOOP 5 */
  for (int m=0; m<nsteps; m++){//This can't just use collapse directive here as some loops inside this loop cannot be parallelised

    #pragma omp parallel default(none) shared(NX,NY,u,bval_left,bval_right,bval_lower,bval_upper)//explicity specifying shared variables
    {
            /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
        /* LOOP 6 */
        for (int j=0; j<NY+2; j++){
        u[0][j]    = bval_left;
        u[NX+1][j] = bval_right;
        }

        /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
        /* LOOP 7 */
        for (int i=0; i<NX+2; i++){
        u[i][0]    = bval_lower;
        u[i][NY+1] = bval_upper;
        }
    }

    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
    //can't parallelise because we have a dependency where we need i-1 elements that could of been executed by other threads causing unexpected results
    for (int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
           if (y[j] <= zZero){//this code adds vertical sheer
            velx = 0;
           }else{
            velx = (velocity_constant*log(y[j]/zZero));
           }
	          dudt[i][j] = -velx * (u[i][j] - u[i-1][j]) / dx
	            - vely * (u[i][j] - u[i][j-1]) / dy;
      }
    }
    
    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
    //i will remove open mp direcvie here or noe and move on 
    #pragma omp parallel for default(none) shared(NX,NY,dt,dudt,u) collapse(2)//we have to perform a reduction as it would give innacuare results because here we are accumulating a sum so it has to be done in a specific manner for the result to be valid
    //unable to use reduction caluse for u as we get a segmentation fault due to the stack growing too large for the program to handle
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
        //no need for reduction as they arrays don't have any overlapping values so the threads can seperately compute each array sums without any race conditions.
	        u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
    
  } // time loop
  
  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  //cant paralise once again because we need ot deliver data in only one order which isn't guranteed by multithreading 
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);

  //creating vertical average we want to plot a average value of u for each x point.
  //must exlude boaundry values
  FILE *verticalaverage;
  verticalaverage = fopen("VerticalAverage.dat", "w");
  /* LOOP 11 */
  float vertical_avg[NX][2]; //a array that holds average values of u for every x value {X,TOTAL,COUNT} total/count will give the average u value for a x point
  int nextfree = 1;
  int found = -1;
  // {X VALUE , total U VALUE,total number of u values }
    for (int i=1; i<NX; i++){// changed the nested loop pointers to exclude the boundary values : 0 and last 
        for (int j=1; j<NY; j++){ 
       //IF FOUND ADD U AND AVG
       found = -1;
        for (int k=1; k<nextfree; k++){ //check if paricular x value was inserted already
            if (vertical_avg[k][0]==x[i]){
                //add to the array a new value of x
                vertical_avg[k][1]= vertical_avg[k][1] + u[i][j];
                vertical_avg[k][2]= vertical_avg[k][2] + 1.0; //increments the count as we added a new value
                found = 1;
                break;//exit loop as its done for this iteration 
                 printf("looped %d",k);
            }
           // printf("looped %d",k);
        }
        if (found == -1){ //only this block is explored....
            //printf("looped %d",1);
            //add new x value into the array
            vertical_avg[nextfree][0] = x[i];
            vertical_avg[nextfree][1] = u[i][j];
            vertical_avg[nextfree][2] = vertical_avg[nextfree][2] + 1.0 ;
            nextfree+=1; //points to next free array space
        }
    }
  }
  //now we need to divide the total sum by number of elements that created the total value of u to find the average u for each x value.
  for (int k= 0 ;k<nextfree-1;k++){
        vertical_avg[k][1] = vertical_avg[k][1] / NX; 
        fprintf(verticalaverage,"%g %g\n", vertical_avg[k][0],vertical_avg[k][1]);
        //prints x value on the x axis and the u value of the y axis.
  }
  fclose(verticalaverage);
  return 0;
} 