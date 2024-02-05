"""Module providing computing functions."""

from typing import List, Union, Optional
import numpy as np

# import scipy.constants as sc
# import library as Lib
import json

# Constants
IDEAL_GAS_MOLAR_VOLUME = 22.4  # liter/mol
AVOGADRO_NUMBER = 6.022e23  # particles/mol
BOLTZMANN_CONSTANT_EV = 8.6173303e-5  # eV/K
BOLTZMANN_CONSTANT_L = 83.14  # mBar*Liter/(mol*K)
GXE_DENSITY = 5.5e-3  # Kg/liter


class Outgassing_setup:
    """
    This class represents a system with various attributes and methods,
    extracting data from a JSON file instead of a Python module.

    Attributes:
    name (str): The name of the setup.
    material (str): The material of the system.
    solute (str): The solute in the system.
    version (str): The version of the system.
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
        # Load data from the JSON file
        with open("library.json", "r") as file:
            data = json.load(file)

        self.name: str = setup
        self.material: Optional[str] = material
        self.solute: Optional[str] = solute
        self.version: Optional[str] = version

        self.temperatures: List[Union[int, float]] = []
        self.diffusion_constants: List[float] = []

        self.time: List[List[float]] = []
        self.impurities: List[List[float]] = []
        self.flow_rates: List[List[float]] = []

        # Retrieve material properties safely from JSON data
        if material and solute:
            material_props = data["Material"].get(material, {}).get(solute, {})
            self.diffusion: Optional[float] = material_props.get("Diffusion Constant")
            self.solubility: Optional[float] = material_props.get("Solubility")
            self.activation_energy: Optional[float] = material_props.get(
                "Activation Energy"
            )

        # Retrieve system properties safely from JSON data
        if setup and material and version:
            system_props = (
                data["System"].get(setup, {}).get(material, {}).get(version, {})
            )
            self.volume: float = system_props.get("Volume")
            self.area: float = system_props.get("Area")
            self.thickness: float = system_props.get("Thickness")

        # Retrieve gas properties safely from JSON data
        if solute:
            gas_props = data["Gas"].get(solute, {})
            self.abundance: Optional[float] = gas_props.get("Abundance in Air")
            self.molar_mass: Optional[float] = gas_props.get("Molar Mass")

        # Retrieve Xenon Mass and Field Factor from JSON data
        self.xe_mass: Optional[float] = data["System"].get(setup, {}).get("Xenon Mass")
        self.field_factor: Optional[float] = (
            data["System"].get(setup, {}).get("Field Factor")
        )

    def __str__(self) -> str:
        """
        Return a string representation of the system with attributes that are not None or empty.
        """
        attributes = vars(self)
        non_empty_attributes = {
            item: attributes[item]
            for item in attributes
            if attributes[item] not in [None, [], ""]
        }
        return "\n".join(
            f"{key}: {value}" for key, value in non_empty_attributes.items()
        )

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

    def get_steel_flow_rate_vs_pumping_time(
        self, unbaked_flow_rate: float, initial_pumped_time: float
    ) -> None:
        """
        Calculate the flow rate over time considering diffusion constants and constraints.
        More versatile law should be implemented from nexo outgassing paper (feb 2020) & https://doi.org/10.1016/S0042-207X(03)00035-6
        """
        if unbaked_flow_rate is None or initial_pumped_time is None:
            raise ValueError(
                "unbaked_flow_rate and initial_pumped_time must be provided."
            )

        # Ensure time attribute is not empty or null
        if not self.time or self.time[0] is None:
            raise ValueError("Time attribute is not set for the system.")

        # Calculate flow rate for each time point
        flow_rate = [
            unbaked_flow_rate * self.area * initial_pumped_time / time
            for time in self.time[0]
        ]

        # Append calculated flow rate to the flow_rates attribute
        self.flow_rates.append(flow_rate)

    def get_electron_lifetime_vs_time(
        self,
        initial_impurities: float,
        circulation_rate: float,
        purification_efficiency: float,
        out_diffusion: float,
        purifier_output: float = 0,
    ) -> List[List[float]]:
        """
        Calculate the electron lifetime for each timestamp in each sublist of self.time.
        """
        if self.xe_mass is None:
            raise ValueError("Xenon mass (xe_mass) is not set.")

        electron_lifetimes = []
        for time_segment in self.time:
            segment_lifetimes = []
            for timestamp in time_segment:
                factor_exp = np.exp(
                    -GXE_DENSITY
                    * purification_efficiency
                    * circulation_rate
                    * timestamp
                    / self.xe_mass
                )
                denominator = initial_impurities * factor_exp + (
                    (out_diffusion + purifier_output * circulation_rate)
                    / (purification_efficiency * circulation_rate)
                ) * (1 - factor_exp)

                if denominator == 0:
                    segment_lifetimes.append(float("inf"))
                else:
                    lifetime = self.field_factor / denominator
                    segment_lifetimes.append(lifetime)

            electron_lifetimes.append(segment_lifetimes)

        return electron_lifetimes


def solve_diffusion_equation(
    time: float, diff: float, thickness: float, conc: float
) -> float:
    """
    Solve the diffusion equation over a given time period.
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
    """
    terms = [
        np.exp(-((np.pi * (2.0 * n + 1.0) / thickness) ** 2) * diff * time)
        * (4.0 * conc * diff)
        / thickness
        for n in range(1000)
    ]
    return sum(terms) * area


def get_time_stamps(
    points: List[Union[int, float]],
    spacing: Union[int, float],
    time_scale: str = "Seconds",
) -> List[List[float]]:
    """
    Generate a list of lists containing timestamps in seconds.
    Each sublist represents the range from one point to the next,
    incremented by the given spacing value.
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


def solve_electron_lifetime(t, M, n0, F, eta, R0, alpha, n_p=0):
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
    factor_exp = np.exp(-GXE_DENSITY * eta * F * t / M)
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
    - M: in kg, LXe mass
    - n0_error (optional): error associated with n0
    - R0_error (optional): error associated with R0

    Returns:
    - Main electron lifetime, and (optionally) lower and upper bounds due to errors
    """
    electron_lifetime = C_el / (n0 + R0 * GXE_DENSITY * t / M)

    if n0_error is not None and R0_error is not None:
        electron_lifetime_upper = C_el / (
            n0 - n0_error + (R0 - R0_error) * GXE_DENSITY * t / M
        )
        electron_lifetime_lower = C_el / (
            n0 + n0_error + (R0 + R0_error) * GXE_DENSITY * t / M
        )
        return electron_lifetime, electron_lifetime_lower, electron_lifetime_upper

    return electron_lifetime
