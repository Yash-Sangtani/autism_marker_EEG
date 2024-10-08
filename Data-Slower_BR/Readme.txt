Data associated with Spiegel et al., Slower Binocular Rivalry in the Autistic Brain.
Alina Spiegel, Jeff Mentch, Amanda J. Haskins, Caroline E. Robertson

***
./FFTData contains frequency-domain data:

-One .mat file / participant / rivalry or simulation block
-Each .mat file contains two variables: elecFFT (Amplitude) and freqAxis (Frequency)
-How to plot? figure; plot(freqAxis,elecFFT). 
-Can recreate figure S1 in Spiegel et al., 2019.

***
./RLSData contains time-domain data:

-One .mat file / participant
-Each .mat file contains a cell array with four cells corresponding to four types of transitions: 
   1. Rivalry trials - left eye to right eye
   2. Rivalry trials - right eye to left eye
   3. Simulation trials - 5.7 Hz to 8.5 Hz
   4. Simulation trials - 8.5 Hz to 5.7 Hz
- Each cell contains structures: 
     ->For rivalry trials: transitions{i}.left and transitions{i}.right
     ->For simulation trials: transitions{i}.f1 and transitions{i}.f2  
-Each row in a structure corresponds to the 5121 timepoints +/- 5 seconds around one reported transition 
-./makeGroupPlot.m recreates Figure 2B in Spiegel et al., 2019 for illustration purposes.