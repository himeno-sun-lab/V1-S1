 # Neural Network Simulation Project

This project implements a neural network simulation framework with support for both local and cluster-based execution.

## Project Structure

- `run.py`: Main entry point for launching simulations
- `sim_S1_M1.py`: Simulation module for the connection S1 and M1 regions
- 
- `nest_routine.py`: Core neural network simulation routines using NEST
- `baseCTXParams.py`, `baseBGParams.py`, `baseTHParams.py`: Base parameter configurations for different brain regions
- `ctxM1Params.py`: Specific parameters for cortical and motor regions
- `simParams.py`: General simulation parameters
- `ini_all.py`: Initialization routines
- `stim_all_model.py`: Stimulation model implementations

## Key Features

- Support for local and cluster-based execution (Sango and K clusters)
- Parameterized neural network simulations
- Modular architecture for different brain regions
- Git-based version tracking for reproducibility
- Flexible parameter configuration system

## Dependencies

- Python 3.x
- NEST neural simulator
- NumPy
- Other dependencies as specified in the code

## Usage

1. Configure simulation parameters in the appropriate parameter files
2. Run simulations using:
   ```bash
   python run.py [options]
   ```

## Directory Structure

- `CBneurons/`: Cerebellar neuron models
- `CBnetwork/`: Cerebellar network implementations
- `ctx/`: Cortical region implementations
- `params/`: Parameter configuration files
- `log/`: Log files
- `code_improvement/`: Code improvement and development files

## Notes

- The project uses a timestamp-based directory naming scheme for experiment results
- Git integration ensures reproducibility of simulations
- Parameter configurations can be overridden via command line arguments

