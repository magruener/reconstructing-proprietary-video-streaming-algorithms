# Understanding video streaming algorithms in the wild

In this repo, you'll find all the data to recreate the results shown in 
the ATC2020 paper "Reconstructing Proprietary Video Streaming Algorithms."

### Folder Structure
```
.
+-- ABRPolicies # Contains Academia ABR approaches
|   +-- PensieveMultiVideo # PensieveMultiBitrate Code
|   +-- PensieveSingleVideo # Pensieve Single Video
|   +-- ABRPolicy.py # Interface
|   +-- ComplexABRPolicy.py # Wrapper for multiple methods
|   +-- OptimalABRPolicy.py # Optimal ABR policy 
|   +-- SimpleABRPolicy.py # Rate based/Buffer based approach
|   +-- ThroughputEstimator.py # Throughput estimators
+-- BehaviourCloning # Cloning Approaches
|   +-- ActionCloning.py # Behavioral cloning wrapper
|   +-- GeneticFeatureEngineering.py #Genetic algorithms
|   +-- ImitationLearning.py # VIPER and DAgger implementation
|   +-- MLABRPolicy.py # Tree-based base function
|   +-- RewardShapingCloning.py # Deep RL Imitation
+-- DashExperiments # Code needed to obtain the graphs comparing to industry
+-- Data ## Can be obtained from https://polybox.ethz.ch/index.php/s/vxDU0robZYwL7WX
|   +-- FeedbackResults # original measurements
|   +-- ParsedResults # parsed original measurements into testing environment
|   +-- Traces # complete trace collection
|   +-- trees_generated_evaluation # best trees for each provider
|   +-- trees_visualised # visualisation for all trees
|   +-- Video_Info # video information for different providers
|   +-- VideoSelectionList # videos selected from different providers
+-- Experiments
|   +-- SampleExperiment.py # Example experiment
+-- SimulationEnviroment
|   +-- Rewards.py # Contains rewards that can be used in the environment
|   +-- SimulatorEnviroment.py # Simulator
+-- Transformer
|   +-- LegacyTransformer.py # Contains transformers which map raw measurements to ParsedResults
|   +-- reference_creation_script.py # Creates evaluation for academia ABRs
|   +-- sample_creation_script.py # Script to parse all measurements
+-- Utility
|   +-- tree_mapper.py #Small tool to map the trees generated in this code to a more readable version
```
### Generating FeedbackResults
The measurements have been generated with the code found in https://github.com/magruener/understanding-video-streaming-in-the-wild
### Tree Interpretation
We have made the best trees we were able to generate for a given leaf number restriction available
in the folder trees_visualised. Here we'll provide a short explanation for the composite features
used in the trees. If you want to adapt the code to use your features, have a look at MLABRPolicy.py
 * Lookahead: How far do we plan. (Only if applicable)
 * Lookback: How many past values do we consider. Depending on the features the 
 lookback can also pertain to just a specific value
 * Buffer Fill / Download Time (s): For each second downloaded how many seconds do we lose/gain
 * (Buffer Fill / Download Time (s)) x Mbit Gain: For each second downloaded how many seconds do we 
 loose/gain weighted by the rate we're obtaining
 * (Buffer Fill / Download Time (s)) x VMAF Gain: For each second downloaded how many seconds do we 
 loose/gain weighted by the VMAF we're obtaining
 * Linear QoE Normalised: QoE as in the MPC paper difference of QoE achieved by this action and the 
  minimum QoE achievable. 
 * Linear QoE: QoE as in the MPC paper raw
 * Buffer Drainage (s): How long did we have to wait for there to be enough space in the buffer so that we can download
 a new chunk
### Prerequisites
The scripts were tested and developed for Python 3.7. 
* Install the python packages needed via requirements.txt

### Example
You can find an example experiment under Experiments 
## Authors

* **Maximilian Grüner** - [ETH Zürich](mailto:mgruener@ethz.ch)
* **Melissa Licciardello** - [ETH Zürich](mailto:melissa.licciardello@inf.ethz.ch)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details

## More Information

For more information, see the version available via the ATC2020 webpage.
