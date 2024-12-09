from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

print("\nQuery1: P(Battery | CarMoves = No)")
print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))

print("\nQuery2: P(CarStarts | Radio = Doesn't turn on)")
print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))

print("\nQuery3: P(Radio | Battery = Works, Gas = Yes) vs P(Radio | Batters = Works)")
prob_radio_w_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
prob_radio_wo_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Empty"})
print(f"With Gas: {prob_radio_w_gas}")
print(f"Without Gas: {prob_radio_wo_gas}")

print("\nQuery5: P(CarStarts | Radio = turns on, Gas = Full")
print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))

cpd_keypresent = TabularCPD(
    variable="KeyPresent",
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]}
)

cpd_starts_updated = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ["yes", "no"],
        "Ignition": ["Works", "Doesn't work"],
        "Gas": ["Full", "Empty"],
        "KeyPresent": ["yes", "no"]
    }
)

car_model.add_node("KeyPresent")
car_model.add_edge("KeyPresent", "Starts")
car_model.remove_cpds(cpd_starts)
car_model.add_cpds(cpd_keypresent, cpd_starts_updated)

assert car_model.check_model()

car_infer = VariableElimination(car_model)

print("\nQuery6: P(KeyPresent | CarMoves = no)")
print(car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"}))

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))


