# simReport.py

# Python libraries
from fpdf import FPDF
import os
import random # For dummy data generation in __main__

# Local Libraries
from simUtils import get_current_date
from simStats import createFilesStats, createQueueStats, createStayStats
from simProcess import SimulationParameters # Import our central parameter class
import pandas as pd # For dummy data generation in __main__
import numpy as np # For dummy data generation in __main__

pageWidth = 210
pageHeight = 297

# Adjust filePathPrefix to be relative to the script's execution directory
# Assumes 'plots' and 'reportResources' are sibling directories to where the main script runs.
filePathPrefix = "." 

reportDate = get_current_date() 

def getFileName(fileName: str, sim_params: SimulationParameters, timePeriod: float):
        """
        Main entry point to create the simulation report.
        
        Args:
            fileName (str): Base name for the output PDF file.
            sim_params (SimulationParameters): The simulation parameters object.
            timePeriod (float): The total simulation time period for analysis.
        """
        global outPutFile, timeWindow

        timeWindow = timePeriod
        filePrefix = fileName+"_" # Prefix for plot image file names
        
        outPutFile = str(fileName)+".pdf"
        create_analytics_report(filePrefix, sim_params, outPutFile, reportDate)

def createTitle(day, pdf):
        """Creates the title page of the PDF report."""
        # Ensure reportResources exists and dtraHeader.png is there
        header_path = os.path.join(filePathPrefix, "reportResources", "dtraHeader.png")
        if os.path.exists(header_path):
            pdf.image(header_path, 0, 0, pageWidth)
        else:
            print(f"Warning: dtraHeader.png not found at {header_path}. Skipping header image.")
            # Optionally add placeholder text if image is missing
            pdf.set_font('Arial', 'B', 18)
            pdf.cell(0, 10, "Simulation Report", 0, 1, "C")
            pdf.ln(5)

        pdf.set_font('Arial', 'B', 14)  
        pdf.cell(0, 5, "Capabilities Development Data Ingestion", 0, 1, "C", False)
        pdf.cell(0, 5, "Process Model", 0, 1, "C", False)
        pdf.ln(2)
        pdf.set_font('Arial', 'I', 10) 
        pdf.cell(0, 5, "Simulation Report", 0, 1, "C", False)
        pdf.ln(12)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 4, "Produced on: "+f'{day}', 0, 1, "R", False)

def createHeader(pdf):
        """Creates the header for content pages."""
        pdf.set_font('Arial', '', 8)  
        pdf.cell(0, 5, "Capabilities Development Data Ingestion Process Model - Simulation Report", 0, 0, "L", False)
        pdf.cell(0, 4, "Page "f'{pdf.page_no()}', 0, 1, "R", False)

def _add_plot_image(pdf, plot_name, x, y, width):
    """Helper to add a plot image if it exists."""
    plot_path = os.path.join(filePathPrefix, "plots", plot_name)
    if os.path.exists(plot_path):
        pdf.image(plot_path, x, y, width)
    else:
        print(f"Warning: Plot image not found at {plot_path}. Skipping image for {plot_name}.")
        # Optionally add a placeholder or text indicating missing plot
        pdf.set_font('Arial', '', 8)
        pdf.text(x, y + width/4, f"Plot Missing: {plot_name}")


def _add_statistics_table(pdf, stats_data):
    """
    Helper to add a formatted statistics table.
    Assumes stats_data is a flat list:
    [file_stats (6), queue_stats (5), stay_stats (5)] = 16 values
    """
    i = 0
    pdf.set_font('Arial', "BU", 8)
    pdf.cell(43, h = 5, txt = "File Statistics", ln = 1, align = "R")
    pdf.set_font('Arial', "", 8)
    pdf.cell(33, h = 3, txt = "Total Files:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Min File per Month:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Max Files per Month:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Median Files per Month:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Mean Files per Month:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Standard Deviation:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1; pdf.ln(5)

    pdf.set_font('Arial', "BU", 8)
    pdf.cell(43, h = 5, txt = "Queue Statistics", ln = 1, align = "R")
    pdf.set_font('Arial', "", 8)
    pdf.cell(33, h = 3, txt = "Min Queue Length:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Max Queue Length:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Median Queue Length:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Mean Queue Length:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Standard Deviation:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1; pdf.ln(5)

    pdf.set_font('Arial', "BU", 8)
    pdf.cell(43, h = 5, txt = "Stay Statistics", ln = 1, align = "R")
    pdf.set_font('Arial', "", 8)
    pdf.cell(33, h = 3, txt = "Min Stay (Days):", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Max Stay (Days):", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Median Stay (Days):", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Mean Stay (Days):", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R"); i+=1
    pdf.cell(33, h = 3, txt = "Standard Deviation:", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{stats_data[i]:10.2f}", ln = 1, align = "R")
    pdf.ln(20)


def _add_file_type_page(pdf, file_type_prefix: str, file_type_name: str, file_prefix: str, time_window: float):
    """Adds a page with statistics and plots for a specific file type."""
    pdf.add_page()
    createHeader(pdf)

    # Note: Assumes file_name for stats is "FILETYPE_Stay" or "FILETYPE_Files"
    # This implies simDataFormat.py names its files consistently.
    file_stats_data = createFilesStats(f"{file_type_prefix.upper()}_Stay", time_window, "tomato")
    queue_stats_data = createQueueStats(f"{file_type_prefix.upper()}_Files") # Assuming these are outputted to separate files
    stay_stats_data = createStayStats(f"{file_type_prefix.upper()}_Stay")
    
    # Concatenate all stats for the generic table function
    all_stats = file_stats_data + queue_stats_data + stay_stats_data

    pdf.set_font('Arial', "BU", 8)
    pdf.cell(15, h = 3, txt = f"{file_type_name} Statistics", ln = 1, align = "L")

    # Plot images (assuming names like FILEPREFIX_FILETYPE_Queue_length_Stair.png)
    # The filePrefix is passed from getFileName, which is like "SYS_" or "SimName_".
    # The statistics functions in simStats create plots named like "SYS_Stay_Box_Files_per_Month.png"
    # or "SYS_Stay_Box.png"
    # We need to ensure consistency in plot naming between simPlot/simStats and simReport.
    
    # Let's assume the plot names are consistent with the original `simReport.py`
    # and that simStats/simPlot will produce these.
    # For now, I'm using the original naming from your snippet.

    _add_plot_image(pdf, f"{file_prefix}{file_type_name}_Queue_length_Stair.png", 10, 20, pageWidth/3-10)
    _add_plot_image(pdf, f"{file_prefix}{file_type_name}_Queue_length_Hist.png", pageWidth/3+5, 20, pageWidth/3-10)
    _add_plot_image(pdf, f"{file_prefix}{file_type_name}_Queue_length_Box.png", 2*(pageWidth/3), 20, pageWidth/3-10)

    # Files per Month Plot (this one is named without the file_prefix in front)
    _add_plot_image(pdf, f"{file_type_prefix.upper()}_Stay_Box_Files_per_Month.png", 100, 70, pageWidth/3-10)

    pdf.ln(60)
    _add_statistics_table(pdf, all_stats)

def _add_parameters_appendix(pdf, sim_params: SimulationParameters):
    """
    Adds an appendix page detailing the simulation parameters used for this run.
    This replaces the hardcoded index-based approach.
    """
    pdf.add_page()
    createHeader(pdf)
    pdf.set_font('Arial', "BU", 10)
    pdf.ln(5)
    pdf.cell(0, 5, "Appendix A - Parameters for This Simulation Run")
    pdf.ln(5)
    pdf.set_font('Arial', "", 8)

    # General Simulation Parameters
    pdf.cell(33, h = 3, txt = "Simulation Time (min): ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.simTime:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Time Unit: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.timeUnit}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Processing FTE: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.processingFte:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Processing Overhead: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.processingOverhead:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Processing Efficiency: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.processingEfficiency:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Warmup Time (min): ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.warmupTime:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Number of Replications: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.numberOfReplications}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Random Seed: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.seed}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "SIPR Transfer Time: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.siprTransferTime:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "File Growth Slope: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.fileGrowth:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Sensor Growth Slope: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.sensorGrowth:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Ingest Efficiency Slope: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.ingestEfficiency:.2f}", ln = 1, align = "L")
    pdf.cell(33, h = 3, txt = "Goal Parameter: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{sim_params.goalParameter}", ln = 1, align = "L")
    pdf.ln(5)

    # Parameters for Each File Type
    pdf.set_font('Arial', "BU", 8)
    pdf.cell(0, 5, "File Type Specific Parameters:", ln = 1, align = "L")
    pdf.set_font('Arial', "", 8)
    
    # Sort file types for consistent report order
    sorted_file_types = sorted(sim_params.file_type_params.items())

    for prefix, params in sorted_file_types:
        pdf.ln(2) # Small line break for clarity between file types
        pdf.set_font('Arial', "B", 8)
        pdf.cell(0, 3, f"--- {params['name']} ---", ln = 1, align = "L")
        pdf.set_font('Arial', "", 8)
        
        pdf.cell(33, h = 3, txt = "  Mean Processing Time: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{params['processing_time']:.2f}", ln = 1, align = "L")
        pdf.cell(33, h = 3, txt = "  Mean Files per Month: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{params['files_per_month']:.2f}", ln = 1, align = "L")
        pdf.cell(33, h = 3, txt = "  Interarrival Time (min): ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{params['iat']:.2f}", ln = 1, align = "L")
        pdf.cell(33, h = 3, txt = "  Batch Size: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{params['batch_size']}", ln = 1, align = "L")
        pdf.cell(33, h = 3, txt = "  Distribution Type: ", ln = 0, align = "R"); pdf.cell(10, h = 3, txt = f"{params['distribution_type']}", ln = 1, align = "L")
        # You can add distribution_kwargs here if desired, but it might get too long
        # pdf.cell(33, h = 3, txt = "  Distribution Kwargs: ", ln = 0, align = "R"); pdf.multi_cell(0, 3, f"{params['distribution_kwargs']}", align = "L")

    pdf.ln(20) # Space before end of page

def create_analytics_report(filePrefix: str, sim_params: SimulationParameters, outPutFile: str, day: str):
        """
        Generates the full PDF analytics report.

        Args:
            filePrefix (str): Prefix for plot image file names.
            sim_params (SimulationParameters): The simulation parameters object.
            outPutFile (str): The name of the output PDF file.
            day (str): The date for the report.
        """
        global pdf # Make pdf global for helper functions like _add_file_type_page to access
        pdf = FPDF() # A4 (210 by 297 mm)

        ''' Title Page '''
        pdf.add_page()
        createTitle(day, pdf)

        ''' Introduction '''
        pdf.set_font('Arial', "B", 10)
        pdf.ln(5)
        pdf.cell(0, 5, "Purpose")
        pdf.ln(8)
        pdf.set_font('Arial', "", 10)
        pdf.multi_cell(pageWidth-20, 4, txt = "     The purpose of this analysis is to evaluate the Phalanx Data Optimization Team's (DOT) capabilities " \
                       "and capacity across different manning levels and demand scenarios. This includes determining the support " \
                        "capabilities and wait times associated with varying manning levels and demand signals. Ultimately, the " \
                        "analysis will identify the optimal DOT manning level needed to handle the assumed maximum demand, while also " \
                        "establishing procurement gates for acquiring additional personnel to accommodate future increases in workload. " \
                        "A key component will be to document all tasks currently performed by the DOT, along with the Full-Time " \
                        "Equivalents (FTEs) required for each, and to identify any additional analytic tasks, and their associated " \
                        "manning requirements, that would be needed if the Phalanx/C-UXS Task Force were expanded.")
        pdf.ln(11)
        pdf.set_font('Arial', "B", 10)
        pdf.cell(0, 5, "Contraints, Limitations & Assuptions")
        pdf.set_font('Arial', "", 10)
        pdf.ln(8)
        pdf.multi_cell(pageWidth-20, 4, 
                        txt = "Constraints:\n - Software limited to that available on DTRA NLAN, SLAN and UNET." \
                                "\n - The file simStart.py must be run in a debugger enabled Visual Studio Code installation, or from a local installation of python IDLE.")
        pdf.ln(5)
        pdf.multi_cell(pageWidth-20, 4, 
                        txt = "Limitations:\n - Parameter value estimations are based on the data available and can be refined as more accurate data is generated." \
                                "\n - The model is designed to identify areas of risk; it is not designed to provide predictive results.")
        pdf.ln(5)
        pdf.multi_cell(pageWidth-20, 4, 
                        txt = "Assumptions:\n - All active sensors (except Windtalker) send one dataset per month. Windtalker sensors send one dataset per week." \
                                "\n - The number of active sensors increases by 20% per year (ceiling of 100% NORTHCOM GPL + 50% OCONUS GPL + 100% ships + 88 sites "\
                                "\n   on the Southern Border = 1,146 sites with up to 1,485 sensors by type)." \
                                "\n - Processing efficiency improves 10% per year (floor of 10 minutes per dataset)." \
                                "\n - Baseline parameter values assume four FTEs dedicated to data ingestion of seven FTEs available to the DOT." \
                                "\n     - Only workdays are simulated." \
                                "\n             - Eight hours per day." \
                                "\n             - Five days per week." \
                                "\n             - 52 weeks per year." \
                                "\n     - Federal holidays (11 at 88 total hours) are explicitlyaccounted for." \
                                "\n     - No sick day or vacation time is accounted for in the model.")
        pdf.ln(20)

        ''' System-wide Statistics Page '''
        pdf.add_page()
        createHeader(pdf)
        
        # Calculate system-wide statistics (assuming "SYS_Stay" and "SYS_Files" are the aggregated names)
        sysFile = createFilesStats("SYS_Stay", timeWindow, "white")
        sysQueue = createQueueStats("SYS_Files")
        sysStay = createStayStats("SYS_Stay")
        sysStats = sysFile+sysQueue+sysStay

        pdf.set_font('Arial', "BU", 8)
        pdf.cell(15, h = 3, txt = "System Statistics", ln = 1, align = "L")

        # System Time in Queue Statistics Plots
        _add_plot_image(pdf, f"{filePrefix}Length_of_Stay_Stair.png", 10, 20, pageWidth/3-10)
        _add_plot_image(pdf, f"{filePrefix}Length_of_Stay_Hist.png", pageWidth/3+5, 20, pageWidth/3-10)
        _add_plot_image(pdf, f"{filePrefix}Length_of_Stay_Box.png", 2*(pageWidth/3), 20, pageWidth/3-10)

        # System Queue Length Statistics Plots
        _add_plot_image(pdf, f"{filePrefix}System_Queue_Length_Stair.png", 10, 70, pageWidth/3-10)
        _add_plot_image(pdf, f"{filePrefix}System_Queue_Length_Hist.png", pageWidth/3+5, 70, pageWidth/3-10)
        _add_plot_image(pdf, f"{filePrefix}System_Queue_Length_Box.png", 2*(pageWidth/3), 70, pageWidth/3-10)

        # Files per Month Plot (system-wide)
        _add_plot_image(pdf, f"SYS_Stay_Box_Files_per_Month.png", 100, 120, pageWidth/3-10)

        pdf.ln(110)
        # System statistics data table
        _add_statistics_table(pdf, sysStats)
        
        ''' Additional Pages for each File Type '''
        # Iterate through active file types from SimulationParameters
        # We'll only generate pages for file types that are active (IAT > 0)
        for prefix, params in sim_params.file_type_params.items():
            if params['iat'] > 0: # Only add page for active file types
                _add_file_type_page(pdf, prefix, params['name'], filePrefix, timeWindow)

        ''' Last Page: Appendix A - Simulation Parameters '''
        _add_parameters_appendix(pdf, sim_params)

        # Output the PDF
        # Ensure the 'reports' directory exists
        report_output_path = os.path.join(filePathPrefix, "reports")
        if not os.path.exists(report_output_path):
            os.makedirs(report_output_path)
        
        pdf.output(os.path.join(report_output_path, outPutFile))
        print(f"Report generated: {outPutFile}")

# Example of how to use this module (for testing purposes, typically called from 01simStart.py)
if __name__ == '__main__':
    print("Running a test report generation from simReport.py __main__ block.")

    # Create dummy directories and files for testing
    if not os.path.exists("./data"): os.makedirs("./data")
    if not os.path.exists("./plots"): os.makedirs("./plots")
    if not os.path.exists("./reportResources"): os.makedirs("./reportResources")
    if not os.path.exists("./reports"): os.makedirs("./reports") # Ensure reports directory exists

    # Create a dummy dtraHeader.png (a small red box)
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (pageWidth, 100), color = 'red')
        d = ImageDraw.Draw(img)
        d.text((10,10), "Dummy Header Image", fill=(255,255,0))
        img.save(os.path.join("./reportResources", "dtraHeader.png"))
    except ImportError:
        print("Pillow not installed. Cannot create dummy header image. Please install with 'pip install Pillow'.")

    # Create dummy data files and plots for testing
    # This requires simDataFormat.py and simPlot.py to be callable.
    # For a quick test, we'll manually create some dummy CSVs and image files.
    
    # Dummy data for SYS_Stay and SYS_Files
    dummy_df_sys_stay = pd.DataFrame({
        "timeStep": list(range(1, 121)),
        "fileNum": list(range(1, 121)),
        "stayLength": np.random.rand(120) * 5 + 1 # Random stay times
    })
    dummy_df_sys_stay.to_csv("./data/SYS_Stay.csv", index=True) # index=True to mimic simDataFormat output

    dummy_df_sys_files = pd.DataFrame({
        "timeStep": list(range(1, 121)),
        "queueLength": np.random.rand(120) * 20 # Random queue lengths
    })
    dummy_df_sys_files.to_csv("./data/SYS_Files.csv", index=True)

    # Dummy data for CO_Stay and CO_Files
    dummy_df_co_stay = pd.DataFrame({
        "timeStep": list(range(1, 121)),
        "fileNum": list(range(1, 121)),
        "stayLength": np.random.rand(120) * 3 + 0.5 # Random stay times
    })
    dummy_df_co_stay.to_csv("./data/CO_Stay.csv", index=True)

    dummy_df_co_files = pd.DataFrame({
        "timeStep": list(range(1, 121)),
        "queueLength": np.random.rand(120) * 10 # Random queue lengths
    })
    dummy_df_co_files.to_csv("./data/CO_Files.csv", index=True)

    # Create dummy plots (simple colored images)
    plot_width = int(pageWidth/3-10)
    plot_height = int(plot_width * 0.75) # Maintain aspect ratio
    
    dummy_plot_names = [
        "SYS_Length_of_Stay_Stair.png", "SYS_Length_of_Stay_Hist.png", "SYS_Length_of_Stay_Box.png",
        "SYS_System_Queue_Length_Stair.png", "SYS_System_Queue_Length_Hist.png", "SYS_System_Queue_Length_Box.png",
        "SYS_Stay_Box_Files_per_Month.png", # This one is named directly in simReport
        
        # For the dynamic page, let's create a dummy for 'CO'
        # Note: The filePrefix is added by getFileName, which is 'TestReport_' in this __main__ block
        "TestReport_CO_Queue_length_Stair.png", "TestReport_CO_Queue_length_Hist.png", "TestReport_CO_Queue_length_Box.png",
        "CO_Stay_Box_Files_per_Month.png" # This one doesn't get the filePrefix
    ]
    
    try:
        for plot_name in dummy_plot_names:
            img = Image.new('RGB', (plot_width, plot_height), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            d = ImageDraw.Draw(img)
            d.text((10,10), f"Dummy {plot_name}", fill=(255,255,255))
            img.save(os.path.join("./plots", plot_name))
    except ImportError:
        print("Pillow not installed. Cannot create dummy plot images. Please install with 'pip install Pillow'.")


    # Mock SimulationParameters for testing
    from simProcess import SimulationParameters # Needs to be imported for the class
    test_params_dict = {
        'sim_time': 120, # 120 minutes for the test
        'co_time': 8.0,
        'co_iat': 1.2, # Active CO sensor
        'dk_time': 15.0,
        'dk_iat': 0.0, # Inactive DK sensor
        'nf1_time': 20.0,
        'nf1_iat': 5.0, # Active NF1 sensor
        'processing_fte': 2.5,
        'file_growth_slope': 0.03
    }
    sim_params_instance = SimulationParameters(test_params_dict)

    # Call the main report generation function
    getFileName("TestReport", sim_params_instance, sim_params_instance.simTime)

    print("\nTest report generation complete. Check 'reports/TestReport.pdf' and 'data', 'plots', 'reportResources' folders.")
    print("Remember to remove dummy files/folders if not needed.")
