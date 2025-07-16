# simArrayHandler.py

# Import necessary modules from our project
from simProcess import SimulationParameters
from simUtils import work_days_per_year # To get the number of work days per year

class ParameterAdjuster:
    """
    A class to handle the adjustment of simulation parameters, including
    setting initial conditions for looping and applying growth factors.
    It operates directly on a SimulationParameters object.
    """
    def __init__(self):
        # We don't store sim_params here, it's passed to methods for explicit modification.
        pass

    def initialize_loop_parameter(self, sim_params: SimulationParameters, days_per_year: float):
        """
        Resets an initial condition for the parameter to be varied based on goalParameter,
        and sets its starting value for a loop.
        Modifies sim_params in place.
        """
        goal_parameter = sim_params.goalParameter
        initial_loop_value = None # This will store the value of the parameter being initialized

        # Helper to calculate IAT from files_per_month
        def calculate_iat(files_per_month):
            if files_per_month > 0:
                # Formula: iat = 1 / ((files * 12) / daysPerYear)
                # This means IAT is in days/arrival, then convert to minutes later if needed.
                # Our SimulationParameters store IAT in minutes.
                # So, IAT_minutes = (daysPerYear / 12 months) / files_per_month * (minutes_per_day)
                # Let's assume daysPerYear is work_days_per_year and it's 250 days/year
                # and that the iat in SimulationParameters is per minute.
                # So, if files_per_month is arrivals per month, and we want IAT in minutes:
                # arrivals_per_day = files_per_month / (days_per_year / 12)
                # iat_days = 1 / arrivals_per_day
                # iat_minutes = iat_days * (8 * 60) # Assuming 8 hour work day
                # Simplified for the original simArrayHandler's logic:
                # If files_per_month is the monthly rate, and iat is the average time between arrivals in minutes
                # files_per_month = (total_minutes_in_work_month / iat_minutes)
                # total_minutes_in_work_month = days_per_year/12 * 8 hours/day * 60 minutes/hour
                # iat_minutes = (days_per_year/12 * 8 * 60) / files_per_month
                
                # Let's stick to the original simArrayHandler's implied conversion for consistency:
                # iat = 1/((files*12)/daysPerYear)
                # This implies iat is in "days" for calculation, then converted to minutes when used in Salabim.
                # For `SimulationParameters`, `iat` is in minutes.
                # So, if files_per_month is given, `iat` in minutes is:
                # (total working minutes in a year / 12) / files_per_month
                # total_working_minutes_per_year = days_per_year * 8 * 60
                
                # Let's assume the original `iat = 1/((files*12)/daysPerYear)` means:
                # If `iat` is in days, then `files_per_month = daysPerYear / (12 * iat_days)`.
                # If `files_per_month` is the input, then `iat_days = daysPerYear / (12 * files_per_month)`.
                # Then, `iat_minutes = iat_days * (8 * 60)` if the original iat was a "work day" unit.
                
                # Given sim_params.iat is in minutes, and files_per_month is monthly:
                # files_per_month = (days_per_year / 12) * (minutes_per_work_day / iat_minutes)
                # iat_minutes = (days_per_year / 12) * (minutes_per_work_day) / files_per_month
                # minutes_per_work_day = 8 * 60 = 480
                iat_minutes = ((days_per_year / 12.0) * 480) / files_per_month
                return iat_minutes
            return 0.0 # If 0 files, infinite IAT

        # Apply initial conditions based on goal_parameter
        if goal_parameter == "Ingestion FTE":
            sim_params.processingFte = 1.0
            initial_loop_value = sim_params.processingFte
        elif goal_parameter == "SIPR to NIPR Transfer Time":
            sim_params.siprTransferTime = 1.0
            initial_loop_value = sim_params.siprTransferTime
        elif goal_parameter == "File Growth Slope":
            sim_params.fileGrowth = 0.0
            initial_loop_value = sim_params.fileGrowth
        elif goal_parameter == "Sensor Growth Slope":
            sim_params.sensorGrowth = 0.0
            initial_loop_value = sim_params.sensorGrowth
        elif goal_parameter == "Ingest Efficiency Slope":
            sim_params.ingestEfficiency = 0.0
            initial_loop_value = sim_params.ingestEfficiency
        elif "Files per Month" in goal_parameter:
            # Handle all "Files per Month" types
            file_type_key = goal_parameter.split(' ')[0].lower() # e.g., 'co', 'dk', 'new-1'
            if file_type_key == 'new-1': file_type_key = 'nf1' # Handle specific mapping
            elif file_type_key == 'new-2': file_type_key = 'nf2'
            elif file_type_key == 'new-3': file_type_key = 'nf3'
            elif file_type_key == 'new-4': file_type_key = 'nf4'
            elif file_type_key == 'new-5': file_type_key = 'nf5'
            elif file_type_key == 'new-6': file_type_key = 'nf6'
            
            if file_type_key in sim_params.file_type_params:
                sim_params.file_type_params[file_type_key]['files_per_month'] = 1.0
                sim_params.file_type_params[file_type_key]['iat'] = calculate_iat(1.0)
                initial_loop_value = sim_params.file_type_params[file_type_key]['files_per_month']
            else:
                print(f"Warning: Unknown file type key '{file_type_key}' for goal parameter '{goal_parameter}'")
        elif goal_parameter == "WT Files per Weekly Batch":
            # This parameter seems to be specific to 'WT' and not a 'files per month' type.
            # Assuming 'meanWt' in original refers to batch size
            if 'wt' in sim_params.file_type_params:
                sim_params.file_type_params['wt']['batch_size'] = 1.0
                initial_loop_value = sim_params.file_type_params['wt']['batch_size']
            else:
                print("Warning: 'WT' file type not found for 'WT Files per Weekly Batch' goal.")
        else:
            print(f"Warning: Unhandled goal parameter for initialization: {goal_parameter}")

        # Store the initial value for the loop in sim_params
        sim_params.initial_loop_parameter_value = initial_loop_value

        # Re-calculate derived parameters like server_time after changes
        sim_params._calculate_derived_parameters()
        
        return sim_params # Return the modified object

    def _get_active_sensor_count(self, sim_params: SimulationParameters):
        """Helper to count how many sensors (file types) currently have arrivals."""
        count = 0
        for params in sim_params.file_type_params.values():
            if params['iat'] > 0:
                count += 1
        return count

    def _apply_file_growth(self, sim_params: SimulationParameters, days_per_year: float):
        """Applies file growth to relevant file types."""
        for prefix, params in sim_params.file_type_params.items():
            # Apply growth if it's the specific goal parameter or for "All Sensors"
            if sim_params.goalParameter == f"{params['name']} Files per Month" or \
               sim_params.goalParameter == "Growth Applied to All Sensors":
                
                if params['files_per_month'] > 0: # Only apply if currently active
                    params['files_per_month'] *= (1 + sim_params.fileGrowth)
                    # Re-calculate IAT based on new files_per_month
                    # iat_minutes = ((days_per_year / 12.0) * 480) / files_per_month
                    params['iat'] = ((days_per_year / 12.0) * 480) / params['files_per_month']
                    
                    # Special handling for WT's meanWt (batch size) if it exists
                    if prefix == 'wt':
                        # Assuming meanWt in original code corresponds to batch_size
                        # Original: meanWt = meanWt * fileGrowth
                        # This implies new_batch_size = old_batch_size * fileGrowth.
                        # This seems unusual for growth, usually it's (1 + growth).
                        # Let's assume it should be (1 + growth) to match other file growth.
                        params['batch_size'] *= (1 + sim_params.fileGrowth)
                        if params['batch_size'] < 1: params['batch_size'] = 1 # Ensure at least 1

        # Re-calculate derived parameters like server_time after changes
        sim_params._calculate_derived_parameters()


    def _apply_sensor_growth(self, sim_params: SimulationParameters, days_per_year: float):
        """Activates dormant sensors based on sensor growth."""
        # This logic needs to be careful: the original code implies activating sensors
        # with iat == 0, one by one.
        
        # Find dormant sensors (iat == 0)
        dormant_sensors = [
            prefix for prefix, params in sim_params.file_type_params.items()
            if params['iat'] == 0
        ]
        
        # If sensorGrowth is positive, activate one dormant sensor if available
        # The original code's logic `sensorsRemain > 0` and then activating based on `goalParameter`
        # or "Growth Applied to All Sensors" is a bit ambiguous.
        # Let's interpret it as: if sensorGrowth > 0, and there are dormant sensors,
        # activate the next available one (e.g., in alphabetical order of prefixes, or as defined).
        # The original code just activates the first one it finds for the specific goal.
        
        # Let's simplify: if sensorGrowth is > 0, it indicates a desire to activate *new* sensors.
        # The original code activates a specific one if it matches the goalParameter,
        # otherwise, it tries to activate "any" if "Growth Applied to All Sensors".
        # This implies a sequential activation.
        
        # For simplicity, let's say if sensorGrowth > 0, and a specific goal parameter matches
        # a dormant sensor, activate that one. If "Growth Applied to All Sensors",
        # activate the first dormant one found.

        if sim_params.sensorGrowth > 0:
            target_prefix = None
            if "Files per Month" in sim_params.goalParameter:
                # Extract file type key from goal parameter
                gp_prefix = sim_params.goalParameter.split(' ')[0].lower()
                if gp_prefix == 'new-1': gp_prefix = 'nf1'
                elif gp_prefix == 'new-2': gp_prefix = 'nf2'
                elif gp_prefix == 'new-3': gp_prefix = 'nf3'
                elif gp_prefix == 'new-4': gp_prefix = 'nf4'
                elif gp_prefix == 'new-5': gp_prefix = 'nf5'
                elif gp_prefix == 'new-6': gp_prefix = 'nf6'

                if gp_prefix in dormant_sensors:
                    target_prefix = gp_prefix
            elif sim_params.goalParameter == "Growth Applied to All Sensors":
                if dormant_sensors:
                    target_prefix = sorted(dormant_sensors)[0] # Activate the first one alphabetically
            
            if target_prefix and target_prefix in sim_params.file_type_params:
                # Activate the sensor with default values
                params = sim_params.file_type_params[target_prefix]
                params['processing_time'] = 90.0 # Default from original code
                params['files_per_month'] = 2.0 # Default from original code
                params['iat'] = ((days_per_year / 12.0) * 480) / params['files_per_month'] # Recalculate IAT
                # If WT, set batch size
                if target_prefix == 'wt':
                    params['batch_size'] = 2.0 # Default from original code
                    # Original also sets devWt, which is not in SimulationParameters yet.
                    # We'll ignore devWt for now unless explicitly needed in SimulationParameters.

        # Re-calculate derived parameters like server_time after changes
        sim_params._calculate_derived_parameters()


    def _apply_ingest_efficiency(self, sim_params: SimulationParameters):
        """Applies ingest efficiency changes to processing times."""
        for prefix, params in sim_params.file_type_params.items():
            if sim_params.goalParameter == f"{params['name']} Files per Month" or \
               sim_params.goalParameter == "Growth Applied to All Sensors":
                
                if params['processing_time'] >= 10 and params['iat'] != 0: # Only if active and sufficient time
                    # Original: coMinutes = coMinutes + (coMinutes * ingestEfficiency)
                    # This means processing time increases with efficiency? This seems counter-intuitive.
                    # Usually, efficiency means LESS time to process.
                    # If ingestEfficiency is a positive "slope", maybe it means "degradation"?
                    # Or if ingestEfficiency is a negative value for improvement.
                    # Let's assume it's meant to *reduce* processing time if positive, or the slope is negative.
                    # If it's "slope", then a positive slope means increase.
                    # The prompt says "ingestEfficiency" which usually implies improvement.
                    # Let's assume it's meant to REDUCE processing time, so we subtract, or invert the slope.
                    # If the user intends for positive slope to make things *harder* (longer time), keep as is.
                    # For now, I'll assume standard efficiency (positive efficiency makes it faster).
                    # If ingestEfficiency is a "slope", it's probably an *increase* in work content.
                    # Let's stick to the original math for now, which implies longer time for positive slope.
                    params['processing_time'] += (params['processing_time'] * sim_params.ingestEfficiency)
        
        # Re-calculate derived parameters like server_time after changes
        sim_params._calculate_derived_parameters()


    def apply_growth_factors(self, sim_params: SimulationParameters, days_per_year: float):
        """
        Applies file growth, sensor growth, and ingest efficiency based on
        the parameters in sim_params.
        Modifies sim_params in place.
        """
        # Ensure calculated parameters are up to date before applying growth
        sim_params._calculate_derived_parameters()

        # Count how many sensors are currently dormant (IAT is 0)
        num_dormant_sensors = len([
            p for p in sim_params.file_type_params.values() if p['iat'] == 0
        ])

        if sim_params.fileGrowth != 0:
            self._apply_file_growth(sim_params, days_per_year)

        if sim_params.ingestEfficiency != 0:
            self._apply_ingest_efficiency(sim_params)
        
        # Sensor growth only applies if there are dormant sensors and the slope is positive
        if sim_params.sensorGrowth != 0 and num_dormant_sensors > 0:
            self._apply_sensor_growth(sim_params, days_per_year)

        # Final re-calculation of derived parameters after all changes
        sim_params._calculate_derived_parameters()

        return sim_params # Return the modified object


# Example usage (for testing this module independently)
if __name__ == '__main__':
    print("Testing simArrayHandler.py (ParameterAdjuster class)")
    
    # Mock SimulationParameters for testing
    class MockSimulationParameters(SimulationParameters):
        def __init__(self, param_dict):
            super().__init__(param_dict)
            self.initial_loop_parameter_value = None # Add this attribute for testing

    # Get work days per year (using a fixed value for independent testing if simUtils isn't fully set up)
    # For actual use, this would come from simUtils
    test_days_per_year = work_days_per_year(federal_holidays=11, mean_vacation_days=10, mean_sick_days=5)
    print(f"Calculated work days per year: {test_days_per_year:.2f}")

    # --- Test 1: Initialize a loop parameter (e.g., Ingestion FTE) ---
    print("\n--- Test 1: Initialize Ingestion FTE ---")
    test_params_init_fte = MockSimulationParameters({
        'sim_time': 100, 'processing_fte': 5.0, 'goal_parameter': 'Ingestion FTE',
        'co_time': 10.0, 'co_iat': 1.0, 'dk_time': 12.0, 'dk_iat': 2.0
    })
    adjuster = ParameterAdjuster()
    adjusted_params_fte = adjuster.initialize_loop_parameter(test_params_init_fte, test_days_per_year)
    print(f"Initial Ingestion FTE: {adjusted_params_fte.processingFte}")
    print(f"Initial Loop Value: {adjusted_params_fte.initial_loop_parameter_value}")

    # --- Test 2: Initialize a loop parameter (e.g., CO Files per Month) ---
    print("\n--- Test 2: Initialize CO Files per Month ---")
    test_params_init_co = MockSimulationParameters({
        'sim_time': 100, 'processing_fte': 1.0, 'goal_parameter': 'CO Files per Month',
        'co_time': 10.0, 'co_iat': 10.0, # Existing values
        'dk_time': 12.0, 'dk_iat': 2.0
    })
    adjusted_params_co = adjuster.initialize_loop_parameter(test_params_init_co, test_days_per_year)
    print(f"CO Files per Month: {adjusted_params_co.file_type_params['co']['files_per_month']:.2f}")
    print(f"CO IAT: {adjusted_params_co.file_type_params['co']['iat']:.2f} minutes")
    print(f"Initial Loop Value: {adjusted_params_co.initial_loop_parameter_value:.2f}")

    # --- Test 3: Apply File Growth ---
    print("\n--- Test 3: Apply File Growth ---")
    test_params_growth = MockSimulationParameters({
        'sim_time': 100, 'processing_fte': 1.0, 'goal_parameter': 'Growth Applied to All Sensors',
        'file_growth_slope': 0.10, # 10% growth
        'co_time': 10.0, 'co_iat': 1.0, 'co_files_per_month': 100.0, # 100 files/month initially
        'dk_time': 12.0, 'dk_iat': 2.0, 'dk_files_per_month': 50.0, # 50 files/month initially
        'wt_time': 5.0, 'wt_iat': 5.0, 'wt_batch_size': 5.0, 'wt_files_per_month': 20.0
    })
    print(f"Initial CO Files: {test_params_growth.file_type_params['co']['files_per_month']:.2f}")
    print(f"Initial DK Files: {test_params_growth.file_type_params['dk']['files_per_month']:.2f}")
    print(f"Initial WT Batch Size: {test_params_growth.file_type_params['wt']['batch_size']:.2f}")

    adjusted_params_growth = adjuster.apply_growth_factors(test_params_growth, test_days_per_year)
    print(f"After Growth CO Files: {adjusted_params_growth.file_type_params['co']['files_per_month']:.2f}")
    print(f"After Growth CO IAT: {adjusted_params_growth.file_type_params['co']['iat']:.2f} minutes")
    print(f"After Growth DK Files: {adjusted_params_growth.file_type_params['dk']['files_per_month']:.2f}")
    print(f"After Growth DK IAT: {adjusted_params_growth.file_type_params['dk']['iat']:.2f} minutes")
    print(f"After Growth WT Batch Size: {adjusted_params_growth.file_type_params['wt']['batch_size']:.2f}")


    # --- Test 4: Apply Sensor Growth (activate a new sensor) ---
    print("\n--- Test 4: Apply Sensor Growth ---")
    test_params_sensor_growth = MockSimulationParameters({
        'sim_time': 100, 'processing_fte': 1.0, 'goal_parameter': 'Growth Applied to All Sensors',
        'sensor_growth_slope': 0.1, # Positive slope to trigger activation
        'co_time': 10.0, 'co_iat': 1.0, # Active
        'dk_time': 12.0, 'dk_iat': 0.0, # Dormant
        'nf1_time': 0.0, 'nf1_iat': 0.0 # Dormant, will be activated next alphabetically
    })
    print(f"Initial DK IAT: {test_params_sensor_growth.file_type_params['dk']['iat']:.2f}")
    print(f"Initial NF1 IAT: {test_params_sensor_growth.file_type_params['nf1']['iat']:.2f}")

    # Activate DK first (alphabetical order of dormant sensors)
    adjusted_params_sensor_growth_1 = adjuster.apply_growth_factors(test_params_sensor_growth, test_days_per_year)
    print(f"After 1st Sensor Growth DK IAT: {adjusted_params_sensor_growth_1.file_type_params['dk']['iat']:.2f}")
    print(f"After 1st Sensor Growth NF1 IAT: {adjusted_params_sensor_growth_1.file_type_params['nf1']['iat']:.2f}")
    print(f"After 1st Sensor Growth DK Proc Time: {adjusted_params_sensor_growth_1.file_type_params['dk']['processing_time']:.2f}")
    print(f"After 1st Sensor Growth DK Files per Month: {adjusted_params_sensor_growth_1.file_type_params['dk']['files_per_month']:.2f}")

    # Reset slope to 0 to prevent accidental re-activation, then set again to activate another
    adjusted_params_sensor_growth_1.sensorGrowth = 0.0
    adjusted_params_sensor_growth_1 = adjuster.apply_growth_factors(adjusted_params_sensor_growth_1, test_days_per_year) # Apply without sensor growth
    adjusted_params_sensor_growth_1.sensorGrowth = 0.1 # Set slope again
    adjusted_params_sensor_growth_2 = adjuster.apply_growth_factors(adjusted_params_sensor_growth_1, test_days_per_year)

    print(f"After 2nd Sensor Growth NF1 IAT: {adjusted_params_sensor_growth_2.file_type_params['nf1']['iat']:.2f}")
    print(f"After 2nd Sensor Growth NF1 Proc Time: {adjusted_params_sensor_growth_2.file_type_params['nf1']['processing_time']:.2f}")
    print(f"After 2nd Sensor Growth NF1 Files per Month: {adjusted_params_sensor_growth_2.file_type_params['nf1']['files_per_month']:.2f}")


    # --- Test 5: Apply Ingest Efficiency ---
    print("\n--- Test 5: Apply Ingest Efficiency ---")
    test_params_ingest = MockSimulationParameters({
        'sim_time': 100, 'processing_fte': 1.0, 'goal_parameter': 'Growth Applied to All Sensors',
        'ingest_efficiency_slope': 0.05, # 5% increase in processing time (original interpretation)
        'co_time': 10.0, 'co_iat': 1.0,
        'dk_time': 12.0, 'dk_iat': 0.0, # Dormant, should not change
    })
    print(f"Initial CO Processing Time: {test_params_ingest.file_type_params['co']['processing_time']:.2f}")
    print(f"Initial DK Processing Time: {test_params_ingest.file_type_params['dk']['processing_time']:.2f}")

    adjusted_params_ingest = adjuster.apply_growth_factors(test_params_ingest, test_days_per_year)
    print(f"After Ingest Efficiency CO Processing Time: {adjusted_params_ingest.file_type_params['co']['processing_time']:.2f}")
    print(f"After Ingest Efficiency DK Processing Time: {adjusted_params_ingest.file_type_params['dk']['processing_time']:.2f}")
