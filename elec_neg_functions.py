"""Module providing computing functions."""
import numpy as np
import scipy.constants as sc
import library as Lib

# Constants
IDEAL_GAS_MOLAR_VOLUME = 22.4  # liter/mol
AVOGADRO_NUMBER = 6.022E23  # particles/mol
BOLTZMANN_CONSTANT_EV = 8.6173303E-5  # eV/K
BOLTZMANN_CONSTANT_L = 83.14  # mBar*Liter/(mol*K)

class System:
    """
    This class represents a system with various attributes and methods.

    Attributes:
    name (str): The name of the setup.
    material (str): The material of the system.
    solute (str): The solute in the system.
    version (str): The version of the system.
    constraints (list): The constraints of the system.
    constraint_index (list): The index of the constraints.
    diffusion (float): The diffusion constant of the solute in the material.
    solubility (float): The solubility of the solute in the material.
    activation_energy (float): The activation energy of the solute in the material.
    abundance (float): The abundance of the solute in the air.
    molar_mass (float): The molar mass of the solute.
    xe_mass (float): The mass of Xenon in the system.
    volume (float): The volume of the system.
    area (float): The area of the system.
    thickness (float): The thickness of the system.
    """

    def __init__(self, setup, material, solute, version):
        self.name = setup
        self.material = material
        self.solute = solute
        self.version = version

        self.constraints = []
        self.constraint_index = []

        self.diffusion = Lib.Material.get(material).get(solute).get('Diffusion Constant')
        self.solubility = Lib.Material.get(material).get(solute).get('Solubility')
        self.activation_energy = Lib.Material.get(material).get(solute).get('Activation Energy')
        self.abundance = Lib.Gas.get(solute).get('Abundance in Air')
        self.molar_mass = Lib.Gas.get(solute).get('Molar Mass')

        self.xe_mass = Lib.System.get(setup).get('Xenon Mass')
        self.volume = Lib.System.get(setup).get(material).get(version).get('Volume')
        self.area = Lib.System.get(setup).get(material).get(version).get('Area')
        self.thickness = Lib.System.get(setup).get(material).get(version).get('Thickness')

    def print_attributes(self):
        """
        This method prints the attributes of the system.
        """
        attributes = vars(self)
        print('\n'.join(f"{item}: {attributes[item]}" for item in attributes))


def ideal_gas_law(N, T):
    return N*sc.k*T

def do_time_conversion(time_scale):
    """
    Convert different time scales to seconds.

    Args:
    time_scale (str): The time scale to be converted 
    ('Seconds', 'Minutes', 'Hours', 'Days', 'Weeks').

    Returns:
    float: The time in seconds.
    """
    time_scales = {
        'Seconds': 1,
        'Minutes': 60.0,
        'Hours': 3600.0,
        'Days': 3600.0 * 24.0,
        'Weeks': 3600.0 * 24.0 * 7.0
    }
    return time_scales.get(time_scale, "Invalid time scale")

def get_parts_conversion(units):
    """
    Convert parts per unit to their respective numerical values.

    Args:
    units (str): The unit to be converted ('ppm', 'ppb', 'ppt').

    Returns:
    float: The numerical value of the units.
    """
    unit_conversions = {
        'ppm': 1E6,
        'ppb': 1E9,
        'ppt': 1E12
    }
    return unit_conversions.get(units, "Invalid unit")

def get_time_stamps(points, spacing, time_scale):
    x = []
    for ii in range(len(points) - 1):
        time = np.linspace(points[ii], points[ii + 1], int((points[ii + 1] - points[ii]) / spacing + 1))
        x.append(time)
    return x

#########################################################################################
#these 2 functions do the same thing, only left one that is more general in output

def get_diff_temp(data, temperatures):
    """
    Calculate the diffusion constant at given temperatures.

    Args:
    data (object): Data object containing the diffusion constant at room temperature (in in cm2.s-1)
                    and the activation energy (in Joules) attributes.
    temperatures (list): List of temperatures in Kelvin for calculation.

    Returns:
    ndarray: Array of diffusion coefficient at given temperatures.
    """
    return [
        data.diffusion * np.exp(data.activation_energy
                                / BOLTZMANN_CONSTANT_EV * ((1.0 / 293.15) - (1.0 / temp)))
        for temp in temperatures
    ]

def arrhenius(D0, Ea, T):
    """
    Calculate the diffusion coefficient using the Arrhenius equation.
    From "Outgassing Model for Electronegative Impurites in Liquid Xenon for nEXO" 26-02-2020.
    
    Parameters:
    - D0: in cm2.s-1, diffusion constant at room temperature
    - Ea: in Joules, activation energy
    - T: in Kelvin, temperature
    
    Returns:
    - Diffusion coefficient D
    """
    return D0 * np.exp(Ea/sc.k * ((1.0/293.15) - (1.0/T)))

#########################################################################################

def get_initial_impurities(data, units):
    """
    Calculate the initial amount of impurities in various units.

    Args:
    data (object): Data object with Volume, Solubility, Abundance, MolarMass, and XeMass attributes.
    units (str): The unit of measurement for the result ('Mass', 'ppm', 'ppb', 'ppt', '#').

    Returns:
    float: The calculated initial impurities in the specified unit.
    """
    impurity_volume = data.volume * data.solubility * data.abundance
    impurity_mass = impurity_volume / IDEAL_GAS_MOLAR_VOLUME * data.molar_mass

    if 'pp' in units:
        impurity_mass = impurity_mass / data.xe_mass * get_parts_conversion(units)
    elif units == '#':
        impurity_mass = impurity_mass / data.molar_mass * AVOGADRO_NUMBER

    return impurity_mass

def electron_lifetime(t, M, rho, n0, F, eta, R0, alpha, n_p=0):
    """
    Calculate the electron lifetime.
    From "Screening for Electronegative Impurities".
    
    Parameters:
    - t: in seconds, time
    - M: in kg, LXe mass
    - rho: in kg/liter, LXe density
    - n0: in ppb, initial impurity concentration
    - F: in liter/sec, xenon gas circulation flow rate
    - eta: purification efficiency
    - R0: in ppb liter/sec, total out-diffusion rate
    - alpha: in s, EXO-200 value, field dependant factor
    - n_p: in ppb, purifier output impurity concentration
    
    Returns:
    - Electron lifetime value
    """
    factor_exp = np.exp(-rho * eta * F * t / M)
    denominator = n0 * factor_exp + ((R0 + n_p * F) / (eta * F)) * (1 - factor_exp)

    # Guard against division by zero
    if denominator == 0:
        return 0

    return alpha / denominator

def XPM_electron_lifetime_fit(t, C_el, n0, R0, rho, M, n0_error=None, R0_error=None):
    """
    Calculate materials test XPM fit for electron lifetime.
    From "Screening for Electronegative Impurities".
    
    Parameters:
    - t: in seconds, time
    - C_el: in ppb/μs
    - n0: in ppb, initial impurity concentration
    - R0: in ppb liter/sec, total out-diffusion rate
    - rho: in kg/liter, LXe density
    - M: in kg, LXe mass
    - n0_error (optional): error associated with n0
    - R0_error (optional): error associated with R0
    
    Returns:
    - Main electron lifetime, and (optionally) lower and upper bounds due to errors
    """
    electron_lifetime = C_el / (n0 + R0 * rho * t / M)
    
    if n0_error is not None and R0_error is not None:
        electron_lifetime_upper = C_el / (n0 - n0_error + (R0 - R0_error) * rho * t / M)
        electron_lifetime_lower = C_el / (n0 + n0_error + (R0 + R0_error) * rho * t / M)
        return electron_lifetime, electron_lifetime_lower, electron_lifetime_upper
    
    return electron_lifetime

def steel_desorption(tau0, E, T, q0, t):
    """
    Calculate the steel desorption rate. From "Outgassing Model for Electronegative Impurites in Liquid Xenon for nEXO" 26-02-2020.
    
    Parameters:
    - tau0: in seconds, pre-exponential factor for relaxation time
    - E: in Joules, activation energy
    - T: in Kelvin, temperature
    - q0: initial absorbed impurities per unit area
    - t: in seconds, time
    
    Returns:
    - Steel desorption rate J
    """
    residence_time = tau0 * np.exp(E / sc.k * T) #the residence time of an impurity at an absorption site
    J = q0 / residence_time * np.exp(-t / residence_time)
    
    return J

def plastics_outgassing(c0, D0, Ea, T, d, t, surface_area=1.0, max_iter=1000):
    """
    Computes the plastic outgassing rate using the diffusion equation solution.

    Parameters:
    - c0: Initial gas concentration in the sample (ppb).
    - D0: Diffusion constant at infinite temperature (cm^2/s).
    - Ea: Activation energy (Joules).
    - T: Temperature (Kelvin).
    - d: Thickness of the sample (cm).
    - t: Time (seconds).
    - surface_area: Surface area of the material (cm^2). Defaults to 1.0.
    - convert_to_mbar: Convert the result to mBar.Liter/s. Defaults to True.
    - max_iter: Maximum number of iterations for the summation (default is 1000).

    Returns:
    - Outgassing rate (J) in the unit of mBar.Liter/s.
    """
    if t < 0:
        raise ValueError("Time 't' must be positive.")

    J = 0
    D = arrhenius(D0, Ea, T)
    print(D)
    for n in range(max_iter):
        term = np.exp(-(sc.pi * (2 * n + 1) / d)**2 * D * t)
        J += term


    J *= (4 * c0 * D) / d
    R = J*sc.k*273*surface_area
    # Convert to mBar.Liter/s if required
    #ATM_TO_MBAR = 1013.25
    #J *= ATM_TO_MBAR * surface_area
    return R

def plastics_outgassing_approximation(c0, D0, Ea, T, t, d=None, surface_area=1.0, mode='short'):
    """
    Compute the plastic outgassing rate using either the short or long time approximation.
    Automatically converts the result to mBar.Liter/s if requested.

    Parameters:
    - c0: Initial gas concentration in the sample (ppb).
    - D0: Diffusion constant at infinite temperature (cm^2/s).
    - Ea: Activation energy (Joules).
    - T: Temperature (Kelvin).
    - t: Time (seconds).
    - d: Thickness (cm), required for the 'long' mode.
    - mode: 'short' or 'long' to specify the approximation.
    - surface_area: Surface area of the material (cm^2). Defaults to 1.0.
    - convert_to_mbar: Convert the result to mBar.Liter/s. Defaults to True.

    Returns:
    - Outgassing rate (J) based on the selected approximation, in the unit of mBar.Liter/s.
    """
    # Check for invalid inputs
    if t < 0:
        raise ValueError("Time 't' must be positive.")
    if mode == 'long' and (d is None or d <= 0):
        raise ValueError("Thickness 'd' must be specified for the 'long' mode.")

    # Calculate the diffusion coefficient
    D = arrhenius(D0, Ea, T)

    # Calculate outgassing rate
    if mode == 'short':
        if t == 0:
            return None
        J = (c0 * np.sqrt(D))/(np.sqrt(sc.pi)*np.sqrt(t))
    elif mode == 'long':
        if d == 0:
            return None
        J = (4 * c0 * D / d) * np.exp(-(sc.pi / d)**2 * D * t)

    # Convert to mBar.Liter/s
    ATM_TO_MBAR = 1013.25
    J *= ATM_TO_MBAR * surface_area

    return J

def solve_diffusion_equation(time, diff, thickness, conc):
    """
    Solve the diffusion equation over a given time period.

    Args:
    time (float): Time over which to solve the equation.
    diff (float): Diffusion coefficient.
    thickness (float): Thickness of the medium.
    conc (float): Initial concentration.

    Returns:
    float: Result of the diffusion equation calculation.
    """
    terms = [
        (1.0 / ((2.0 * n + 1.0) ** 2))
                                * np.exp(-(np.pi * (2.0 * n + 1.0) / thickness) ** 2 * diff * time)
                                * (conc * 8.0 * thickness) / (np.pi ** 2 * 2.0)
        for n in range(1000)
    ]
    return sum(terms)

def get_flow_rate(time, diff, thickness, conc, area):
    """
    Calculate the flow rate of a substance through a medium.

    Args:
    time (float): Time over which to calculate the flow rate.
    diff (float): Diffusion coefficient.
    thickness (float): Thickness of the medium.
    conc (float): Concentration of the substance.
    area (float): Cross-sectional area of the flow.

    Returns:
    float: Calculated flow rate.
    """
    terms = [
        np.exp(-(np.pi * (2.0 * n + 1.0) / thickness) ** 2 * diff * time)
                                    * (4.0 * conc * diff) / thickness
        for n in range(3)
    ]
    return sum(terms) * area

def get_impurities_vs_time(data, time_scale='Seconds'):
    """
    Calculate the impurities over time considering diffusion constants and constraints.

    Args:
    data (object): Data object containing InitialImpurities,
                                    DiffConstants, Time, Thickness, and Constraints.
    time_scale (str): Time scale for the diffusion equation ('Seconds',
                                    'Minutes', 'Hours', 'Days', 'Weeks').

    Returns:
    ndarray: Array of impurities over time.
    """
    impurities = []
    initial_impurities = data.initial_impurities

    for ii, diff_constant in enumerate(data.diffusion_constants):
        time_segment = np.array(data.time[ii])
        if ii > 0:
            time_segment -= np.max(data.time[ii - 1])

        diff_constant *= do_time_conversion(time_scale)
        y = solve_diffusion_equation(time_segment, diff_constant,
                                    data.thickness, initial_impurities)

        if ii < len(data.constraints):
            y_index = np.where(y < y[0] / data.constraints[ii])[0]
            if len(y_index) > 0:
                y_index = y_index[0]
                y[y_index:] = y[0] / data.constraints[ii]
                data.constraint_index.append(y_index)

        impurities.append(y)
        initial_impurities = y[-1]

    # Adjusting impurities for subsequent time segments
    for ii in range(1, len(impurities)):
        impurities[ii] *= impurities[ii - 1][-1] / impurities[ii][0]

    return np.array(impurities)

def get_flow_rate_vs_time(data, units='#', time_scale='Seconds'):
    """
    Calculate the flow rate over time considering diffusion constants and constraints.

    Args:
    data (object): Data object with Impurities, Volume, Time,
                                DiffConstants, Thickness, Area, ConstraintIndex, and Temp.
    units (str): Units for the flow rate calculation ('#' or 'mBar Liter').
    time_scale (str): Time scale for the calculations ('Seconds',
                                'Minutes', 'Hours', 'Days', 'Weeks').

    Returns:
    ndarray: Array of flow rates over time.
    """
    flow_rate = []
    initial_concentration = [x[0] / (data.volume * 1E3) for x in data.impurities]

    for ii, (time, diff_constant) in enumerate(zip(data.time, data.diffusion_constants)):
        if ii > 0:
            time = time - np.max(data.time[ii - 1])

        diff_constant *= do_time_conversion(time_scale)
        y = get_flow_rate(time, diff_constant, data.thickness, initial_concentration[ii], data.area)
        y /= do_time_conversion(time_scale)

        if ii < len(data.constraint_index):
            y_index = data.constraint_index[ii]
            y_frac = int(y_index // 2.0)
            x_index = np.linspace(0, len(y[y_index // 3:]), len(y[y_frac:]))
            y[y_frac:] = y[y_frac] * np.exp(-0.7E-2 * x_index) + y[y_index]

        flow_rate.append(y)

    if units == 'mBar Liter':
        for ii, y in enumerate(flow_rate):
            flow_rate[ii] = y * (BOLTZMANN_CONSTANT_L * data.temperatures[ii]) / AVOGADRO_NUMBER

    return np.asarray(flow_rate)

def do_modelling(systems, labels, temperature, time, time_scale, constraints=None):
    """
    This function performs modelling for a given set of systems.

    Parameters:
    systems (list): List of systems.
    labels (list): List of labels for the systems.
    temperature (list): List of temperatures.
    time (float): Time for the modelling.
    time_scale (str): Time scale for the modelling.
    constraints (list, optional): List of constraints for the systems. Defaults to None.

    Returns:
    None
    """
    if constraints is None:
        constraints = []

    for i, system in enumerate(systems):
        # Define the different temperatures for which to calculate outgassing
        system.temperatures = temperature

        # Calculate the diffusion constants for the above temperatures using the Arrhenius equation
        system.diffusion_constants = get_diff_temp(system, temperatures=system.temperatures)

        # Get the initial number of impurities from model parameters.
        # Can define '#', 'ppm,ppb,ppt' or 'Mass' for units
        system.initial_impurities = get_initial_impurities(system, '#')

        # Forward above defined labels, that will be later used for the plot legend.
        system.labels = labels[i]

        system.time = time
        if len(constraints) > i:
            system.constraints = constraints[i]

        # Calculate impurities left in the sample vs time using the diffusion equation.
        system.impurities = get_impurities_vs_time(data=system, time_scale=time_scale)

        # Calculate the outgassing rate vs time using Fick's 1st law
        system.flow_rate = get_flow_rate_vs_time(data=system,
                                                 units='mBar Liter', time_scale=time_scale)


def get_labels(systems, temperature):
    """
    This function generates labels for a given set of systems and temperatures.

    Parameters:
    systems (list): List of systems.
    temperature (list): List of temperatures.

    Returns:
    labels (list): List of labels for each system at each temperature.
    """
    labels = []
    for system in systems:
        label = []
        for temp in temperature:
            label.append(rf'{system.version}, $ d= {system.thickness} \, \mathrm{{cm}}, \, T = {
                temp} \, \mathrm{{K}}, \, E_a = {system.activation_energy}\,\mathrm{{eV}}$')
        labels.append(label)
    return labels
