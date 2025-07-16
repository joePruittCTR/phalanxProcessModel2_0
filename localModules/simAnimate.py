import salabim as sb
import random 
# Import the canonical SimulationParameters and the setup helper from the core simulation logic
import simProcess 

# Define a specific Customer class for animation to give it a visual representation
class AnimatedCustomer(simProcess.Customer):
    # ... (rest of AnimatedCustomer class, no changes needed here) ...
    def setup(self, handler_resource):
        super().setup(handler_resource) 
        self.animation_object = sb.Animate3d(
            'sphere.fbx', 
            x=self.env.q.x - 2, y=self.env.q.y, z=self.env.q.z, 
            scale=0.1, color='green',
            parent=self.env.animation_root 
        )

    def process(self):
        yield self.animate(
            x=self.env.q.x, y=self.env.q.y, z=self.env.q.z,
            duration=0.5, 
            _component=self.animation_object
        )

        self.service_start_time = self.env.now()
        self.handler_resource.request() 

        yield self.animate(
            x=self.handler_resource.x, y=self.handler_resource.y, z=self.handler_resource.z,
            duration=0.5, 
            _component=self.animation_object,
        )

        yield self.hold(self.env.sim_params.serverTime) 

        self.handler_resource.release() 
        self.env.customer_service_monitor.tally(self.env.now() - self.service_start_time) 
        self.leave(self.env.q) 

        self.env.customer_stay_monitor.tally(self.life_time()) 

        yield self.animate(
            x=self.handler_resource.x + 2, y=self.handler_resource.y, z=self.handler_resource.z,
            duration=0.5,
            _component=self.animation_object,
            color='red' 
        )
        self.animation_object.remove() 


class AnimatedSource(simProcess.Source):
    # ... (rest of AnimatedSource class, no changes needed here) ...
    pass 


def run_animated_simulation(sim_params: simProcess.SimulationParameters):
    """
    Runs an animated simulation based on the provided SimulationParameters.

    Args:
        sim_params (sim_process.SimulationParameters): An object containing all simulation parameters.

    Returns:
        tuple: A tuple containing two lists of Salabim Monitor objects:
               - fileMonitorList: Monitors related to file processing (e.g., service time).
               - stayMonitorList: Monitors related to total time customers spend in the system.
    """
    # 1. Setup the Salabim Environment for animation
    env = sb.Environment(trace=False, animate=True) 
    # FIX: Removed env.yieldless(False) from here
    
    # FIX: Set yieldless state globally using sb.yieldless()
    sb.yieldless(False)
    # print(f"DEBUG: In simAnimate.py, yieldless set to: {env.is_yieldless()}") # This line will still fail.

    env.random_seed(sim_params.seed)

    # 2. Set up components using the flexible helper function from 03simProcess.py
    sim_objects = simProcess._setup_simulation_components(
        env,
        sim_params,
        customer_class=AnimatedCustomer,
        source_class=AnimatedSource
    )

    # Retrieve the Salabim objects we need for animation from the returned dictionary
    handler_resource = sim_objects['handler_resource']
    customer_queue = sim_objects['customer_queue']
    customer_stay_monitor = sim_objects['customer_stay_monitor']
    customer_service_monitor = sim_objects['customer_service_monitor']

    # 3. Add Animation Elements
    # Set coordinates for visual layout
    customer_queue.x = 2 
    customer_queue.y = 0 
    customer_queue.z = 0 

    handler_resource.x = 5 
    handler_resource.y = 0 
    handler_resource.z = 0 

    # Queue visualization
    sb.AnimateQueue(customer_queue, x=customer_queue.x, y=customer_queue.y, id='customer_queue_vis',
                    spec='box', 
                    fill='lightblue', linecolor='blue',
                    text_offsetx=-0.5, text_offsety=-0.5,
                    title='Customer Queue', text_font_size=10,
                    x_content_gap=0.1, y_content_gap=0.1
                    )

    # Server visualization
    sb.AnimateResource(handler_resource, x=handler_resource.x, y=handler_resource.y, id='server_vis',
                       spec='cube', 
                       color='grey',
                       text_color='black',
                       text_anchor='n',
                       title='Server',
                       text_font_size=10
                       )

    # Monitor displays
    sb.AnimateMonitor(customer_queue.length, x=0, y=3, text_color='black', text_anchor='nw',
                      title='Queue Length', fmt='.0f', text_font_size=12) 
    sb.AnimateMonitor(customer_stay_monitor, x=0, y=2.5, text_color='black', text_anchor='nw',
                      title='Avg Stay Time', fmt='.2f', unit='minutes', text_font_size=12) 
    sb.AnimateMonitor(customer_service_monitor, x=0, y=2, text_color='black', text_anchor='nw',
                      title='Avg Service Time', fmt='.2f', unit='minutes', text_font_size=12) 

    # Static text labels
    sb.AnimateText(x=0, y=3.5, text='Simulation Dashboard', text_color='darkblue', font_size=16, text_anchor='nw')
    sb.AnimateText(x=0, y=1.5, text=f'Sim Time: {sim_params.simTime} min', text_color='gray', font_size=10, text_anchor='nw')
    sb.AnimateText(x=0, y=1.2, text=f'FTEs: {sim_params.processingFte}', text_color='gray', font_size=10, text_anchor='nw')

    # Access IAT and processing time from the 'co' file type parameters
    # Make sure 'co' exists in your sim_params.file_type_params    
    sb.AnimateText(x=0, y=0.9, text=f"IAT (CO): {sim_params.file_type_params['co']['iat']:.2f} min", text_color='gray', font_size=10, text_anchor='nw')
    sb.AnimateText(x=0, y=0.6, text=f"CO Time: {sim_params.file_type_params['co']['processing_time']:.2f} min", text_color='gray', font_size=10, text_anchor='nw')

    # 4. Run Simulation
    env.run(till=sim_params.simTime)

    # 5. Return Monitors
    fileMonitorList = [customer_service_monitor]
    stayMonitorList = [customer_stay_monitor]

    return fileMonitorList, stayMonitorList


# Example of how to use this module (for testing purposes, typically called from 01simStart.py)
if __name__ == '__main__':
    print("Running an animated test simulation from 04simAnimate.py __main__ block.")
    test_params_dict = {
        'sim_time': 200, 
        'time_unit': 'minutes',
        'co_time': 8.0, 'co_iat': 1.5, 
        'processing_fte': 1.0,
        'processing_overhead': 0.10, 'processing_efficiency': 0.90,
        'warmup_time': 0, 'num_replications': 1, 'seed': 42
    }

    sim_params_instance = simProcess.SimulationParameters(test_params_dict)

    service_monitors, stay_monitors = run_animated_simulation(sim_params_instance)

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
