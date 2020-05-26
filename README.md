# Understanding video streaming algorithms in the wild

In this repo, you'll find all the data to recreate the results shown in 
the ATC2020 paper "Reconstructing proprietary video streaming algorithms."

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
+-- Data ## Can be obtained from https://polybox.ethz.ch/index.php/s/EPHqp8yqWRUmUPw
|   +-- FeedbackResults # original measurements
|   |   +-- Arte # Provider
|   |   +-- video_051493-028-A_file_id_report.2011-01-06_0749CET.log_Norway # Video_Trace
|   |   |   +-- local_client_state_logger.csv # Logging of the html5 states
|   |   |   +-- raw_dataframe.csv # Video chunk requests with timing
|   |   |   +-- throttle_logging.tc # Emulated bandwidth over time
|   +-- ParsedResults # parsed original measurements into testing environment
|   |   +-- Arte # Provider
|   |   |   +-- Online # Type of Algorithm
|   |   |   |   +-- evaluation_list # scoring of different runs
|   |   |   |   +-- trajectory_list # trajectory of different runs
|   +-- Traces # complete trace collection
|   +-- trees_generated_evaluation # best trees for each provider with corresponding configuration
|   +-- trees_visualised # visualisation for 10 best trees for each provider
|   |   +-- achieved_score_trees.csv # Contains scoring for trees
|   +-- Video_Info # video information for different providers
|   |   +-- Arte_Online_Complex_Manual_Feature_Engineering_BC_Full__10_leaf_nodes.png ### Provider _ FeatureComplexity _ Method _ Number of leafs
|   +-- VideoSelectionList # videos selected from different providers
+-- Experiments
|   +-- SampleExperiment.py # Example experiment
+-- SimulationEnviroment
|   +-- Rewards.py # Contains rewards functions
|   +-- SimulatorEnviroment.py # Simulator
+-- Transformer
|   +-- LegacyTransformer.py # Contains transformers which map raw measurements to ParsedResults
|   +-- reference_creation_script.py # Creates evaluation for academia ABRs
|   +-- sample_creation_script.py # Script to parse all measurements
+-- Utility
|   +-- tree_mapper.py #Small tool to map the trees generated in this code to a more readable version
```
### Trace Format
Traces are provided in Data/Traces. Each line in the file contains 2 numbers (time, bandwidth), 
where time is in seconds from the start of the script and the bandwidth in Mbps.
### Running the real world DASH experiments
Look at [zdf_tree_server.py](DashExperiments/src/video_server/zdf_tree_server.py) to see how to add your tree to the real world 
evaluation for the video used in [Pensieve](https://github.com/hongzimao/pensieve)
### Generating FeedbackResults
You can regenerate the measurements with the code found in the repository for [Understanding video streaming in the wild](
https://github.com/magruener/understanding-video-streaming-in-the-wild).
### Tree Interpretation
We have made the best trees we were able to generate for a given leaf number restriction available
in the folder trees_visualised. Here we'll provide a short explanation for the composite features
used in the trees. If you want to adapt the code to use your features, have a look at [MLABRPolicy.py](BehaviourCloning/MLABRPolicy.py)
 * Lookahead: How far do we plan. (Only if applicable)
 * Lookback: How many past values do we consider. Depending on the features the 
 lookback can also pertain to just a specific value
 * Buffer Fill / Download Time (s): For each second downloaded how many seconds do we lose/gain
 * (Buffer Fill / Download Time (s)) x Mbit Gain: For each second downloaded how many seconds do we 
 loose/gain weighted by the rate we're obtaining
 * (Buffer Fill / Download Time (s)) x VMAF Gain: For each second downloaded how many seconds do we 
 loose/gain weighted by the VMAF score we're obtaining
 * Linear QoE Normalised: QoE as in the MPC paper, value calculates as the difference of QoE achieved by this action and the 
  minimum QoE achievable. 
 * Linear QoE: QoE as in the MPC paper
 * Buffer Drainage (s): How long did we have to wait for there to be enough space in the buffer so that we can download
 a new chunk
### Incorporating your features
You can manipulate the file [MLABRPolicy.py](BehaviourCloning/MLABRPolicy.py). Be sure to modify the method which
* modifies the feature names
* generates the values
* calculates the number of features
* [tree_mapper.py](Utility/tree_mapper.py) if you want to parse the tree after learning
### Prerequisites
The scripts were tested and developed for Python 3.7 under Ubuntu. 
* Install the python packages needed via requirements.txt

### Example
You can find an example experiment under Experiments 
## Authors

* **Maximilian Grüner** - [ETH Zürich](mailto:mgruener@ethz.ch)
* **Melissa Licciardello** - [ETH Zürich](mailto:melissa.licciardello@inf.ethz.ch)
* **Ankit Singla - [ETH Zürich](mailto:ankit.singla@inf.ethz.ch)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details

## More Information

For more information, see the version available via the ATC2020 webpage.
