import salabim as sb
import random

# Import the Distribution class from our new simDistributions module
from simDistributions import Distribution

class SimulationParameters:
    # ... (rest of SimulationParameters class, no changes needed here) ...
    def __init__(self, param_dict: dict):
        # ... (parameters initialization) ...
        self.file_type_params = {}
        # ... (file_type_params initialization) ...
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
    
    # FIX: Set yieldless state globally using sb.yieldless()
    sb.yieldless(False)
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
