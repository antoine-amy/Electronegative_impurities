"""Module providing computing functions."""

from typing import List, Union, Optional
import numpy as np
import scipy.constants as sc
import library as Lib

# Constants
IDEAL_GAS_MOLAR_VOLUME = 22.4  # liter/mol
AVOGADRO_NUMBER = 6.022e23  # particles/mol
BOLTZMANN_CONSTANT_EV = 8.6173303e-5  # eV/K
BOLTZMANN_CONSTANT_L = 83.14  # mBar*Liter/(mol*K)


class Outgassing_setup:
    """
    This class represents a system with various attributes and methods.

    Attributes:
    name (str): The name of the setup.
    material (str): The material of the system.
    solute (str): The solute in the system.
    version (str): The version of the system.
    constraints (list): The constraints of the system.
    constraint_index (list): The index of the constraints.
    temperatures (list): The temperatures of the system.
    time (list of list of floats): Timestamps of the system.
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

    def __init__(
        self,
        setup: str,
        material: Optional[str] = None,
        solute: Optional[str] = None,
        version: Optional[str] = None,
    ):
        self.name: str = setup
        self.material: Optional[str] = material
        self.solute: Optional[str] = solute
        self.version: Optional[str] = version

        self.constraints: List = []
        self.constraint_index: List[int] = []
        self.temperatures: List[Union[int, float]] = []
        self.time: List[List[float]] = []
        self.diffusion_constants: List[float] = []

        self.impurities: List[List[float]] = []
        self.flow_rates: List[List[float]] = []

        # Retrieve material properties safely
        if material and solute:
            material_props = Lib.Material.get(material, {}).get(solute, {})
            self.diffusion: Optional[float] = material_props.get("Diffusion Constant")
            self.solubility: Optional[float] = material_props.get("Solubility")
            self.activation_energy: Optional[float] = material_props.get(
                "Activation Energy"
            )

        # Retrieve system properties safely
        if setup and material and version:
            system_props = Lib.System.get(setup, {}).get(material, {}).get(version, {})
            self.volume: float = system_props.get("Volume")
            self.area: float = system_props.get("Area")
            self.thickness: float = system_props.get("Thickness")

        # Retrieve gas properties safely
        if solute:
            gas_props = Lib.Gas.get(solute, {})
            self.abundance: Optional[float] = gas_props.get("Abundance in Air")
            self.molar_mass: Optional[float] = gas_props.get("Molar Mass")

        # Retrieve Xenon Mass
        self.xe_mass: Optional[float] = Lib.System.get(setup, {}).get("Xenon Mass")

    def print_attributes(self) -> None:
        """
        This method prints the attributes of the system.
        """
        attributes = vars(self)
        for item in attributes:
            print(f"{item}: {attributes[item]}")

    def get_diff_temp(self) -> None:
        """
        Calculate the diffusion constant at the instance's temperatures.
        """
        if self.diffusion is None or self.activation_energy is None:
            raise ValueError("Diffusion or activation energy not set.")

        self.diffusion_constants = [
            self.diffusion
            * np.exp(
                self.activation_energy
                / BOLTZMANN_CONSTANT_EV
                * ((1.0 / 293.15) - (1.0 / temp))
            )
            for temp in self.temperatures
        ]

    def get_initial_impurities(self, units: str) -> None:
        """
        Calculate the initial amount of impurities in various units.
        """
        if (
            self.volume is None
            or self.solubility is None
            or self.abundance is None
            or self.molar_mass is None
            or self.xe_mass is None
        ):
            raise ValueError(
                "System attributes not fully set for impurity calculation."
            )

        impurity_volume = self.volume * self.solubility * self.abundance
        impurity_mass = impurity_volume / IDEAL_GAS_MOLAR_VOLUME * self.molar_mass

        # Unit conversion and impurity_mass calculation
        if "pp" in units:
            conversion_factor = (
                1e6
                if units == "ppm"
                else 1e9 if units == "ppb" else 1e12 if units == "ppt" else 1
            )
            self.initial_impurities /= self.xe_mass * conversion_factor
        elif units == "#":
            self.initial_impurities = (
                impurity_mass / self.molar_mass
            ) * AVOGADRO_NUMBER

    def get_impurities_vs_time(self) -> None:
        """
        Calculate the impurities over time considering diffusion constants and constraints.
        """
        if self.diffusion_constants is None:
            raise ValueError("Diffusion constants are not initialized.")

        # Use the same time segment for all diffusion constants if only one time sublist
        time_segments = (
            [self.time[0]] * len(self.diffusion_constants)
            if len(self.time) == 1
            else self.time
        )

        for diff_constant, time_segment in zip(self.diffusion_constants, time_segments):
            segment_impurities = []
            current_impurity = (
                self.initial_impurities
            )  # Local variable for current impurity

            for timestamp in time_segment:
                current_impurity = solve_diffusion_equation(
                    timestamp, diff_constant, self.thickness, current_impurity
                )
                segment_impurities.append(current_impurity)

            self.impurities.append(segment_impurities)

    def get_flow_rate_vs_time(self, units: str = "#") -> None:
        """
        Calculate the flow rate over time considering diffusion constants and constraints.
        """
        initial_concentration = [x[0] / (self.volume * 1e3) for x in self.impurities]

        time_segments = (
            [self.time[0]] * len(self.diffusion_constants)
            if len(self.time) == 1
            else self.time
        )

        for diff_constant, temp, time_segment in zip(
            self.diffusion_constants, self.temperatures, time_segments
        ):
            segment_flow_rates = []
            for timestamp in time_segment:
                flow_rate = solve_flow_rate(
                    timestamp,
                    diff_constant,
                    self.thickness,
                    initial_concentration[0],
                    self.area,
                )

                if units == "mBar Liter":
                    flow_rate *= (BOLTZMANN_CONSTANT_L * temp) / AVOGADRO_NUMBER

                segment_flow_rates.append(flow_rate)

            self.flow_rates.append(segment_flow_rates)


def solve_diffusion_equation(
    time: float, diff: float, thickness: float, conc: float
) -> float:
    """
    Solve the diffusion equation over a given time period.

    Args:
    - time (float): Time over which to solve the equation.
    - diff (float): Diffusion coefficient.
    - thickness (float): Thickness of the medium.
    - conc (float): Initial concentration.

    Returns:
    - float: Result of the diffusion equation calculation.
    """
    terms = [
        (1.0 / ((2.0 * n + 1.0) ** 2))
        * np.exp(-((np.pi * (2.0 * n + 1.0) / thickness) ** 2) * diff * time)
        * (conc * 8.0 * thickness)
        / (np.pi**2 * 2.0)
        for n in range(1000)
    ]
    return sum(terms)


def solve_flow_rate(
    time: float, diff: float, thickness: float, conc: float, area: float
) -> float:
    """
    Calculate the flow rate of a substance through a medium.

    Args:
    - time (float): Time over which to calculate the flow rate.
    - diff (float): Diffusion coefficient.
    - thickness (float): Thickness of the medium.
    - conc (float): Concentration of the substance.
    - area (float): Cross-sectional area of the flow.

    Returns:
    - float: Calculated flow rate.
    """
    terms = [
        np.exp(-((np.pi * (2.0 * n + 1.0) / thickness) ** 2) * diff * time)
        * (4.0 * conc * diff)
        / thickness
        for n in range(1000)
    ]
    return sum(terms) * area


def get_time_stamps(points, spacing, time_scale="Seconds"):
    """
    Generate a list of lists containing timestamps in seconds.
    Each sublist represents the range from one point to the next,
    incremented by the given spacing value.

    Args:
    points (list of float): Time points, in the specified time scale, between which timestamps are generated.
    spacing (float): Increment value in seconds.
    time_scale (str, optional): Unit of time for points ('Days', 'Hours', 'Seconds'). Defaults to 'Seconds'.

    Returns:
    list of list of float: Lists of timestamps in seconds.
    """
    scale_factors = {"Days": 86400, "Hours": 3600, "Seconds": 1}

    if time_scale not in scale_factors:
        raise ValueError(
            "Unsupported time scale. Choose from 'Days', 'Hours', or 'Seconds'."
        )

    converted_points = [point * scale_factors[time_scale] for point in points]

    timestamps = []
    for start_point, end_point in zip(converted_points, converted_points[1:]):
        current_time = start_point
        time_segment = []
        while current_time < end_point:
            time_segment.append(current_time)
            current_time += spacing * scale_factors[time_scale]
        timestamps.append(time_segment)

    return timestamps


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
    residence_time = tau0 * np.exp(
        E / sc.k * T
    )  # the residence time of an impurity at an absorption site
    J = q0 / residence_time * np.exp(-t / residence_time)

    return J
