import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load and plot frequency-domain data (FFTData)
def plot_fft_data(mat_file, label='FFT Data Visualization'):
    try:
        data = sio.loadmat(mat_file)
        elecFFT = data.get('elecFFT')
        freqAxis = data.get('freqAxis')
        
        if elecFFT is not None and freqAxis is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(freqAxis.squeeze(), elecFFT.squeeze(), color='b')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(label)
            plt.grid(True)
            plt.show()
        else:
            print(f"Variables 'elecFFT' and 'freqAxis' not found in {mat_file}.")
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")

# Load and plot time-domain data (RLSData)
def plot_rls_data(mat_file):
    try:
        data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        transitions = data.get('transitions')
        
        if transitions is not None:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            titles = [
                "Rivalry: Left to Right Eye",
                "Rivalry: Right to Left Eye",
                "Simulation: 5.7 Hz to 8.5 Hz",
                "Simulation: 8.5 Hz to 5.7 Hz"
            ]
            
            for i, ax in enumerate(axs.flatten()):
                if i < len(transitions):
                    transition_data = transitions[i]

                    # Plot rivalry transitions
                    if i < 2:
                        left_data = getattr(transition_data, 'left', None)
                        right_data = getattr(transition_data, 'right', None)

                        if left_data is not None and right_data is not None:
                            ax.plot(left_data, label='Left Eye', color='r')
                            ax.plot(right_data, label='Right Eye', color='g')
                    # Plot simulation transitions
                    else:
                        f1_data = getattr(transition_data, 'f1', None)
                        f2_data = getattr(transition_data, 'f2', None)

                        if f1_data is not None and f2_data is not None:
                            ax.plot(f1_data, label='5.7 Hz', color='m')
                            ax.plot(f2_data, label='8.5 Hz', color='c')
                
                    ax.set_title(titles[i])
                    ax.set_xlabel('Time (ms)')
                    ax.set_ylabel('Amplitude')
                    ax.legend(loc='upper right')
                    ax.grid(True)
                else:
                    ax.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print(f"Variable 'transitions' not found in {mat_file}.")
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")

# Example usage: replace with actual file paths
plot_fft_data('./FFTData/cumulus01_Rival1.mat', label='FFT Data Visualization - Rival 1')
plot_fft_data('./FFTData/cumulus01_Sim1.mat', label='FFT Data Visualization - Sim 1')
plot_rls_data('./RLSData/cumulus02.mat')
