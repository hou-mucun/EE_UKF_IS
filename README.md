# Description

## Information

This repository contains figrue-generating scripts in Python for the student lab rotation project:

"Evaluation and Extension of an Unscented Kalman Filter for Estimating Insulin Sensitivity in Type 1 Diabetes"

## Requirement

- Python (3.6)
  
- Python packages: numpy (1.19.5), scipy (1.5.4), pandas (1.1.5), seaborn (0.9.0), matplotlib (3.3.3), filterpy (1.4.5)

## Author

Mucun Hou <mucuhou@ethz.ch>

## Usage

`Trail(npop, sim_dur, ts = 1, BW = 70, Gb = 110, IS_nom = 2.47e-5/0.015, m2_nom = 0.0513, meal = {'time': [7, 12, 18], 'D': [60e3, 80e3, 70e3]},exercise = {'time': [14, 16], 'intensity': 4317}, extra_CHO = {'time': 16, 'D': 15e3}, basal_insulin_time = 22)`

    Create a virtual trail consiting of a population of T1D patients with uniform parameters except for insulin sensitivity.

    properties: npop: int
                    The number of virtual patients.
                sim_dur: int
                    The total period of trail [h]
                ts: int, optional
                    Time of sampling
                BW: int, optional
                    Body weight
                Gb: int, optional
                    Target blood glucose
                IS_nom: double, optional
                    Nominal insulin sensitivity
                m2_nom: double, optional
                    Nominal glucose appearance rate from meals. In simulations it is sample in Gaussian distribution
                meal: dicionary, optional
                    The meal time and carbonhydrate amount
                    key     type            unit    comment
                    'time'  list of int     hour    breakfast, lunch, dinner
                    'D'     list of double  mg      breakfast, lunch, dinner
                exercise: dictionary, optional
                    The exercise time and intensity
                    key         type            unit    comment
                    'time'      list of int     hour    start_time, end_time
                    'intensity' int             -       accelerometer account
                extra_CHO: dicionary, optional
                    The time and carbonhydrate amount of extra food after exercise
                    key     type    unit    comment
                    'time'  int     hour    -
                    'D'     int     mg      -
                basal_insulin_time: int, optional
                    The time for basal insulin infusion adjustment

## Examples

As in `Fig1_evaluation.py`, `Fig2_extension.py`, `Fig3_force_drop.py`, `Fig4_mpc.py`, `Fig5_scenarios.py`.
