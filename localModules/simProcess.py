import salabim as sb
import random
# FIX: Set yieldless state globally using sb.yieldless()
sb.yieldless(False)

# Import the Distribution class from our new simDistributions module
from simDistributions import Distribution

class SimulationParameters:
    """
    Centralized class for defining and holding all simulation parameters.
    It maps input parameters from 02simInput.py's format to internal, consistent names.
    Now supports multiple file types/sensors with custom arrival distributions.
    """
    def __init__(self, param_dict: dict):
        # General Simulation Parameters
        self.simTime = param_dict.get('sim_time', 480) # Default 480 minutes (8 hours)
        self.timeUnit = param_dict.get('time_unit', 'minutes')
        self.processingFte = param_dict.get('processing_fte', 1.0)
        self.processingOverhead = param_dict.get('processing_overhead', 0.15)
        self.processingEfficiency = param_dict.get('processing_efficiency', 0.85)
        self.warmupTime = param_dict.get('warmup_time', 0)
        self.numberOfReplications = param_dict.get('num_replications', 1)
        self.seed = param_dict.get('seed', 123) # <-- THIS LINE (and the ones above it) MUST BE PRESENT

        # --- Parameters for Multiple File Types/Sensors ---
        self.file_type_params = {}

        # Define all expected file types and their default values
        file_type_prefixes = [
            'co', 'dk', 'ma', 'nj', 'rs', 'sv', 'tg', 'wt', # Existing types
            'nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6'         # New types
        ]
        
        # Add a placeholder for SIPR transfer time
        self.siprTransferTime = param_dict.get('sipr_transfer_time', 1.0)

        # Parameters for growth/efficiency (from simArrayHandler)
        self.fileGrowth = param_dict.get('file_growth_slope', 0.0)
        self.sensorGrowth = param_dict.get('sensor_growth_slope', 0.0)
        self.ingestEfficiency = param_dict.get('ingest_efficiency_slope', 0.0)
        
        # Goal-seeking parameter
        self.goalParameter = param_dict.get('goal_parameter', 'None')

        # Define default distribution kwargs for each type for robustness
        # These are used if specific kwargs are not provided in the input param_dict
        default_dist_kwargs = {
            "MonthlyMixedDist": {"num_days": 30, "first_peak_fraction": 0.05, "second_peak_fraction": 0.10, "first_peak_probability": 0.50, "second_peak_probability": 0.25},
            "WeeklyExponential": {"num_days": 7, "first_day_probability": 0.90},
            "MixedWeibull": {"w1": 0.6, "w2": 0.2, "w3": 0.2, "weibull_shape": 0.8, "weibull_scale": 1.0, "norm_mu": 5, "norm_sigma": 1, "expon_lambda": 0.5},
            "BimodalExpon": {"lambda1": 0.5, "lambda2": 0.5, "weight1": 3/5, "loc2": 15},
            "BetaDistribution": {"alpha": 2, "beta": 5},
            "Exponential": {} # Default Salabim exponential (no extra kwargs needed for our wrapper)
        }

        for prefix in file_type_prefixes:
            # Get raw values from the input param_dict
            proc_time = param_dict.get(f'{prefix}_time', 10.0)
            iat_val = param_dict.get(f'{prefix}_iat', 1.0)
            files_per_month = param_dict.get(f'{prefix}_files_per_month', 0.0)
            batch_size = param_dict.get(f'{prefix}_batch_size', 1)
            
            # Get distribution type and kwargs
            distribution_type = param_dict.get(f'{prefix}_distribution_type', 'Exponential') # Default to Exponential
            
            # Get specific kwargs for this file type's distribution, or use defaults
            dist_kwargs = param_dict.get(f'{prefix}_distribution_kwargs', {})
            # Merge with default kwargs for the chosen distribution type
            final_dist_kwargs = default_dist_kwargs.get(distribution_type, {}).copy()
            final_dist_kwargs.update(dist_kwargs) # User-provided kwargs override defaults

            self.file_type_params[prefix] = {
                'name': prefix.upper(),
                'processing_time': proc_time,
                'iat': iat_val,
                'files_per_month': files_per_month,
                'batch_size': batch_size,
                'distribution_type': distribution_type,
                'distribution_kwargs': final_dist_kwargs,
                'server_time': 0.0 # Will be calculated
            }
        
        # --- Calculated Parameters (for all file types) ---
        self._calculate_derived_parameters()
    def _calculate_derived_parameters(self):
        # ... (derived parameters calculation) ...
        pass

    def get_file_type_names(self):
        return list(self.file_type_params.keys())


class Customer(sb.Component):
    # ... (rest of Customer class, no changes needed here) ...
    def setup(self, handler_resource, file_type: str):
        self.handler_resource = handler_resource
        self.file_type = file_type 
        self.enter(self.env.q) 
        self.service_start_time = None 

    def process(self):
        self.service_start_time = self.env.now()
        self.handler_resource.request() 
        
        specific_server_time = self.env.sim_params.file_type_params[self.file_type]['server_time']
        yield self.hold(specific_server_time) 
        
        self.handler_resource.release() 
        self.env.customer_service_monitor.tally(self.env.now() - self.service_start_time) 
        self.leave(self.env.q) 
        self.env.customer_stay_monitor.tally(self.life_time()) 


class Source(sb.Component):
    # ... (rest of Source class, no changes needed here) ...
    def setup(self, handler_resource, file_type: str, customer_type=Customer):
        self.handler_resource = handler_resource
        self.file_type = file_type
        self.customer_type = customer_type 

    def process(self):
        file_params = self.env.sim_params.file_type_params[self.file_type]
        specific_iat = file_params['iat']
        batch_size = file_params['batch_size']
        distribution_type = file_params['distribution_type']
        distribution_kwargs = file_params['distribution_kwargs']

        if specific_iat == 0: 
            yield self.hold(self.env.infinity) 
            return 

        dist_instance = Distribution(
            mean_interarrival_time=specific_iat, 
            batch_size=1, 
            distribution_type=distribution_type,
            **distribution_kwargs
        )
            
        hold_time = dist_instance.get_interarrival_time() * batch_size 

        yield self.hold(hold_time)
            
        for _ in range(batch_size):
            self.customer_type(handler_resource=self.handler_resource, file_type=self.file_type)


def _setup_simulation_components(env: sb.Environment, sim_params: SimulationParameters,
                                 customer_class=Customer, source_class=Source):
    # ... (rest of _setup_simulation_components, no changes here) ...
    env.sim_params = sim_params 
    env.random_seed(sim_params.seed)

    num_servers = int(sim_params.processingFte)
    handler_resource = sb.Resource(name='handler_resource', capacity=num_servers)

    env.customer_stay_monitor = sb.Monitor(name='Customer_Stay_Time', weight_legend='minutes')
    env.customer_service_monitor = sb.Monitor(name='Customer_Service_Time', weight_legend='minutes')

    env.q = sb.Queue('Customer_Queue')

    for prefix, params in sim_params.file_type_params.items():
        if params['iat'] > 0: 
            source_class(
                handler_resource=handler_resource,
                file_type=prefix, 
                customer_type=customer_class
            )

    return {
        'handler_resource': handler_resource,
        'customer_queue': env.q,
        'customer_stay_monitor': env.customer_stay_monitor,
        'customer_service_monitor': env.customer_service_monitor
    }


def runSimulation(sim_params: SimulationParameters):
    """
    Runs a single simulation based on the provided SimulationParameters.
    This version creates its own environment and runs it.

    Args:
        sim_params (SimulationParameters): An object containing all simulation parameters.

    Returns:
        tuple: A tuple containing two lists of Salabim Monitor objects:
               - fileMonitorList: Monitors related to file processing (e.g., service time).
               - stayMonitorList: Monitors related to total time customers spend in the system.
    """
    # 1. Setup the Salabim Environment
    # FIX: Removed env.yieldless(False) from here
    env = sb.Environment(trace=False)
    
    # print(f"DEBUG: In simProcess.py, yieldless set to: {env.is_yieldless()}") # This line will still fail because env.is_yieldless() is a method, not a property.

    # 2. Set up components using the helper function (uses default Customer and Source classes)
    _setup_simulation_components(env, sim_params)

    # 3. Run Simulation
    env.run(till=sim_params.simTime)

    # 4. Return Monitors
    fileMonitorList = [env.customer_service_monitor]
    stayMonitorList = [env.customer_stay_monitor]

    return fileMonitorList, stayMonitorList


def dataFileGenerator(sim_params: SimulationParameters):
    # ... (rest of dataFileGenerator function, no changes needed here) ...
    print(f"dataFileGenerator called with sim_params.simTime: {sim_params.simTime} minutes")
    print("This function currently does nothing significant and might be moved to 08simDataFormat.py later.")
    pass

# Example of how to use this module (for testing purposes, typically called from 01simStart.py)
if __name__ == '__main__':
    print("Running a test simulation from 03simProcess.py __main__ block.")
    
    test_params_dict = {
        'sim_time': 500,
        'time_unit': 'minutes',
        'processing_fte': 1.0,
        'processing_overhead': 0.10,
        'processing_efficiency': 0.90,
        'warmup_time': 30,
        'num_replications': 1,
        'seed': 42,
        'co_time': 8.0, 'co_iat': 1.2, 
        'dk_time': 15.0, 'dk_iat': 3.0, 'dk_distribution_type': 'MonthlyMixedDist', 'dk_distribution_kwargs': {'num_days': 20, 'first_peak_probability': 0.7}, 
        'nf1_time': 20.0, 'nf1_iat': 0.0, 'nf1_files_per_month': 0.0, 
        'wt_time': 5.0, 'wt_iat': 5.0, 'wt_batch_size': 5, 'wt_distribution_type': 'WeeklyExponential',
        'file_growth_slope': 0.05, 'sensor_growth_slope': 0.0, 'ingest_efficiency_slope': 0.02, 'goal_parameter': 'None'
    }

    sim_params_instance = SimulationParameters(test_params_dict)

    print("\n--- Simulation Parameters ---")
    print(f"Sim Time: {sim_params_instance.simTime} {sim_params_instance.timeUnit}")
    print(f"Processing FTE: {sim_params_instance.processingFte}")
    print(f"File Growth Slope: {sim_params_instance.fileGrowth}")
    print(f"Goal Parameter: {sim_params_instance.goalParameter}")

    print("\n--- File Type Parameters ---")
    for file_type, params in sim_params_instance.file_type_params.items():
        print(f"  {file_type.upper()}:")
        print(f"    Processing Time: {params['processing_time']:.2f} min")
        print(f"    Server Time: {params['server_time']:.2f} min")
        print(f"    IAT: {params['iat']:.2f} min")
        print(f"    Files per Month: {params['files_per_month']:.2f} min")
        print(f"    Batch Size: {params['batch_size']}")
        print(f"    Distribution: {params['distribution_type']}")
        print(f"    Distribution Kwargs: {params['distribution_kwargs']}")


    service_monitors, stay_monitors = runSimulation(sim_params_instance)

    if service_monitors:
        service_time_monitor = service_monitors[0]
        print(f"\nSimulation Results (Service Time):")
        print(f"  N: {service_time_monitor.n}")
        print(f"  Mean Service Time: {service_time_monitor.mean():.2f} minutes")
        print(f"  Max Service Time: {service_time_monitor.maximum():.2f} minutes")

    if stay_monitors:
        stay_time_monitor = stay_monitors[0]
        print(f"\nSimulation Results (Total Stay Time in System):")
        print(f"  N: {stay_time_monitor.n}")
        print(f"  Mean Stay Time: {stay_time_monitor.mean():.2f} minutes")
        print(f"  Max Stay Time: {stay_time_monitor.maximum():.2f} minutes")

    dataFileGenerator(sim_params_instance)
