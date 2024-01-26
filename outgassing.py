import numpy as np

# Constants
IDEAL_GAS_MOLAR_VOLUME = 22.4  # liter/mol
AVOGADRO_NUMBER = 6.022E23  # particles/mol
BOLTZMANN_CONSTANT_EV = 8.6173303E-5  # eV/K
BOLTZMANN_CONSTANT_L = 83.14  # mBar*Liter/(mol*K)

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

def get_diff_temp(data, temperatures):
    """
    Calculate the diffusion temperature difference.

    Args:
    data (object): Data object containing Diffusion and ActivationEnergy attributes.
    temperatures (list): List of temperatures for calculation.

    Returns:
    ndarray: Array of diffusion temperature differences.
    """
    return [
        data.Diffusion * np.exp(data.ActivationEnergy
                                / BOLTZMANN_CONSTANT_EV * ((1.0 / 293.15) - (1.0 / temp)))
        for temp in temperatures
    ]

def get_initial_impurities(data, units):
    """
    Calculate the initial amount of impurities in various units.

    Args:
    data (object): Data object with Volume, Solubility, Abundance, MolarMass, and XeMass attributes.
    units (str): The unit of measurement for the result ('Mass', 'ppm', 'ppb', 'ppt', '#').

    Returns:
    float: The calculated initial impurities in the specified unit.
    """
    impurity_volume = data.Volume * data.Solubility * data.Abundance
    impurity_mass = impurity_volume / IDEAL_GAS_MOLAR_VOLUME * data.MolarMass

    if 'pp' in units:
        impurity_mass = impurity_mass / data.XeMass * get_parts_conversion(units)
    elif units == '#':
        impurity_mass = impurity_mass / data.MolarMass * AVOGADRO_NUMBER

    return impurity_mass

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
    initial_impurities = data.InitialImpurities

    for ii, diff_constant in enumerate(data.DiffConstants):
        time_segment = np.array(data.Time[ii])
        if ii > 0:
            time_segment -= np.max(data.Time[ii - 1])

        diff_constant *= do_time_conversion(time_scale)
        y = solve_diffusion_equation(time_segment, diff_constant,
                                    data.Thickness, initial_impurities)

        if ii < len(data.Constraints):
            y_index = np.where(y < y[0] / data.Constraints[ii])[0]
            if len(y_index) > 0:
                y_index = y_index[0]
                y[y_index:] = y[0] / data.Constraints[ii]
                data.ConstraintIndex.append(y_index)

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
    initial_concentration = [x[0] / (data.Volume * 1E3) for x in data.Impurities]

    for ii, (time, diff_constant) in enumerate(zip(data.Time, data.DiffConstants)):
        if ii > 0:
            time = time - np.max(data.Time[ii - 1])

        diff_constant *= do_time_conversion(time_scale)
        y = get_flow_rate(time, diff_constant, data.Thickness, initial_concentration[ii], data.Area)
        y /= do_time_conversion(time_scale)

        if ii < len(data.ConstraintIndex):
            y_index = data.ConstraintIndex[ii]
            y_frac = int(y_index // 2.0)
            x_index = np.linspace(0, len(y[y_index // 3:]), len(y[y_frac:]))
            y[y_frac:] = y[y_frac] * np.exp(-0.7E-2 * x_index) + y[y_index]

        flow_rate.append(y)

    if units == 'mBar Liter':
        for ii, y in enumerate(flow_rate):
            flow_rate[ii] = y * (BOLTZMANN_CONSTANT_L * data.Temp[ii]) / AVOGADRO_NUMBER

    return np.asarray(flow_rate)
